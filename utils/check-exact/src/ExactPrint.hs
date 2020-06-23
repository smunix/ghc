{-# LANGUAGE DeriveDataTypeable   #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiWayIf           #-}
{-# LANGUAGE NamedFieldPuns       #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE StandaloneDeriving   #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE ViewPatterns         #-}

module ExactPrint
  (
    ExactPrint(..)
  , exactPrint
  -- , exactPrintWithOptions
  , showGhc
  ) where

import GHC
-- import GHC.Hs.Exact
-- import GHC.Hs.Extension
-- import GHC.Parser.Lexer (AddApiAnn(..))
import GHC.Types.Basic hiding (EP)
-- import GHC.Types.Name.Reader
import GHC.Types.SrcLoc
import GHC.Utils.Outputable hiding ( (<>) )

import Control.Monad.Identity
import Control.Monad.RWS
import Data.Data ( Data )
import Data.Foldable
import Data.List ( partition, intercalate)
import Data.Maybe (fromMaybe)
-- import Data.Ord (comparing)

import qualified Data.Map as Map
import qualified Data.Set as Set

-- import qualified GHC

import Lookup
import Utils
import Types

-- import Debug.Trace

-- ---------------------------------------------------------------------

exactPrint :: ExactPrint ast => Located ast -> ApiAnns -> String
exactPrint ast anns = runIdentity (runEP anns stringOptions (exact ast))

type EP w m a = RWST (PrintOptions m w) (EPWriter w) EPState m a
type EPP a = EP String Identity a

runEP :: ApiAnns -> PrintOptions Identity String
      -> Annotated () -> Identity String
runEP anns epReader action =
  fmap (output . snd) .
    (\next -> execRWST next epReader (defaultEPState anns))
    . xx $ action

xx :: Annotated () -> EP String Identity ()
-- xx :: Annotated() -> RWST (PrintOptions m w) (EPWriter w) EPState m ()
xx = id

-- ---------------------------------------------------------------------

defaultEPState :: ApiAnns -> EPState
defaultEPState as = EPState
             { epPos    = (1,1)
             , epAnns   = Map.empty
             , epApiAnns = as
             , epAnnKds = []
             , epLHS    = 0
             , epMarkLayout = False
             , priorEndPosition = (1,1)
             , epComments = rogueComments as
             }


-- ---------------------------------------------------------------------
-- The EP monad and basic combinators

-- | The R part of RWS. The environment. Updated via 'local' as we
-- enter a new AST element, having a different anchor point.
data PrintOptions m a = PrintOptions
            {
              epAnn :: !Annotation
            , epAstPrint :: forall ast . Data ast => GHC.Located ast -> a -> m a
            , epTokenPrint :: String -> m a
            , epWhitespacePrint :: String -> m a
            , epRigidity :: Rigidity
            , epContext :: !AstContextSet
            }

-- | Helper to create a 'PrintOptions'
printOptions ::
      (forall ast . Data ast => GHC.Located ast -> a -> m a)
      -> (String -> m a)
      -> (String -> m a)
      -> Rigidity
      -> PrintOptions m a
printOptions astPrint tokenPrint wsPrint rigidity = PrintOptions
             {
               epAnn = annNone
             , epAstPrint = astPrint
             , epWhitespacePrint = wsPrint
             , epTokenPrint = tokenPrint
             , epRigidity = rigidity
             , epContext = defaultACS
             }

-- | Options which can be used to print as a normal String.
stringOptions :: PrintOptions Identity String
stringOptions = printOptions (\_ b -> return b) return return NormalLayout

data EPWriter a = EPWriter
              { output :: !a }

instance Monoid w => Semigroup (EPWriter w) where
  (<>) = mappend

instance Monoid w => Monoid (EPWriter w) where
  mempty = EPWriter mempty
  (EPWriter a) `mappend` (EPWriter b) = EPWriter (a <> b)

data EPState = EPState
             { epPos    :: !Pos -- ^ Current output position
             , epAnns   :: !Anns
             , epApiAnns :: !ApiAnns
             , epAnnKds :: ![[(KeywordId, DeltaPos)]] -- MP: Could this be moved to the local statE w mith suitable refactoring?
             , epMarkLayout :: Bool
             , epLHS :: LayoutStartCol
             , priorEndPosition :: !Pos -- ^ Position reached when
                                        -- processing the last element
             , epComments :: ![Comment]
             }

-- ---------------------------------------------------------------------

-- AZ:TODO: this can just be a function :: (ApiAnn' a) -> Entry
class HasEntry ast where
  fromAnn :: ast -> Entry

-- ---------------------------------------------------------------------

-- type Annotated = FreeT AnnotationF Identity
type Annotated a = EP String Identity a

-- ---------------------------------------------------------------------

-- | Key entry point.  Switches to an independent AST element with its
-- own annotation, calculating new offsets, etc
markAnnotated :: ExactPrint a => a -> Annotated ()
markAnnotated a = enterAnn (getApiAnnotation a) a

data Entry = Entry RealSrcSpan [RealLocated AnnotationComment]
           | NoEntryVal

instance (HasEntry (ApiAnn' ann)) =>  HasEntry (SrcSpanAnn' (ApiAnn' ann)) where
  fromAnn (SrcSpanAnn ApiAnnNotUsed ss) = Entry (realSrcSpan ss) []
  fromAnn (SrcSpanAnn ann _) = fromAnn ann

instance HasEntry (ApiAnn' a) where
  fromAnn (ApiAnn anchor _ cs) = Entry anchor cs
  fromAnn ApiAnnNotUsed = NoEntryVal

-- | "Enter" an annotation, by using the associated 'anchor' field as
-- the new reference point for calculating all DeltaPos positions.
enterAnn :: (ExactPrint a) => Entry -> a -> Annotated ()
enterAnn NoEntryVal a = do
  p <- getPos
  debugM $ "enterAnn:NO ANN:p =" ++ show p
  exact a
enterAnn (Entry anchor cs) a = do
  printComments anchor
  p <- getPos
  debugM $ "enterAnn:(anchor(pos),p)=" ++ show (ss2pos(anchor),p)
  -- do all the machinery of advancing to the anchor, with a local etc
  -- modelled on exactpc (which is normally called via withast

  -- First thing is to calculate the entry DeltaPos.  This is based on
  -- the current position, and the anchor.
  -- off <- gets apLayoutStart
  off <- gets epLHS
  priorEndAfterComments <- getPos
  let ss = anchor
  let edp = adjustDeltaForOffset
              -- Use the propagated offset if one is set
              -- Note that we need to use the new offset if it has
              -- changed.
              off (ss2delta priorEndAfterComments ss)

  let
    st = annNone { annEntryDelta = edp }
  withOffset st (advance edp >> exact a)

-- ---------------------------------------------------------------------

-- Temporary function to simply reproduce the "normal" pretty printer output
withPpr :: (Outputable a) => a -> Annotated ()
withPpr a = printString False (showGhc a)

-- ---------------------------------------------------------------------
-- Modeled on Outputable

-- | An AST fragment with an annotation must be able to return the
-- requirements for nesting another one, captured in an 'Entry', and
-- to be able to use the rest of the exactprint machinery to print the
-- element.  In the analogy to Outputable, 'exact' plays the role of
-- 'ppr'.
class ExactPrint a where
  getApiAnnotation :: a -> Entry
  exact :: a -> Annotated ()

-- ---------------------------------------------------------------------

-- | Bare Located elements are simply stripped off without further
-- processing.
instance (ExactPrint a) => ExactPrint (Located a) where
  getApiAnnotation (L _ a) = getApiAnnotation a
  exact (L _ a) = exact a

-- ---------------------------------------------------------------------

-- | 'Located (HsModule GhcPs)' corresponds to 'ParsedSource'
instance ExactPrint HsModule where
  getApiAnnotation hsmod = fromAnn (hsmodAnn hsmod)

  exact hsmod@(HsModule ApiAnnNotUsed _ _ _ _ _ _) = withPpr hsmod
  exact (HsModule anns@(ApiAnn ss as cs) mmn mexports imports decls mdeprec mbDoc) = do

    case mmn of
      Nothing -> return ()
      Just (L ln mn) -> do
        markApiAnn anns AnnModule
        debugM $ "HsModule name: (ss,ln)=" ++ show (ss2pos ss,ss2pos (realSrcSpan ln))
        printStringAtSs ln (moduleNameString mn)

        -- forM_ mdeprec markLocated

        forM_ mexports markAnnotated

        markApiAnn anns AnnWhere

    -- markOptional GHC.AnnOpenC -- Possible '{'
    -- markManyOptional GHC.AnnSemi -- possible leading semis
    -- setContextLevel (Set.singleton TopLevel) 2 $ markListWithLayout imports
    markListWithLayout imports

    -- setContextLevel (Set.singleton TopLevel) 2 $ markListWithLayout decls
    markListWithLayout decls

    -- markOptional GHC.AnnCloseC -- Possible '}'

    -- markEOF
    eof <- getEofPos
    printStringAtKw' eof ""


-- ---------------------------------------------------------------------
-- Start of utility functions
-- ---------------------------------------------------------------------

printStringAtSs :: SrcSpan -> String -> EPP ()
printStringAtSs ss str = printStringAtKw' (realSrcSpan ss) str

-- ---------------------------------------------------------------------

-- printStringAtKw :: ApiAnn' ann -> AnnKeywordId -> String -> EPP ()
-- printStringAtKw ApiAnnNotUsed _ str = printString True str
-- printStringAtKw (ApiAnn anchor anns _cs) kw str = do
--   case find (\(AddApiAnn k _) -> k == kw) anns of
--     Nothing -> printString True str
--     Just (AddApiAnn _ ss) -> printStringAtKw' ss str

printStringAtMkw :: Maybe RealSrcSpan -> String -> EPP ()
printStringAtMkw (Just r) s = printStringAtKw' r s
printStringAtMkw Nothing s = printStringAtLsDelta [] (DP (0,1)) s

printStringAtKw' :: RealSrcSpan -> String -> EPP ()
printStringAtKw' ss str = do
  printComments ss
  dp <- nextDP ss
  p <- getPos
  debugM $ "printStringAtKw': (dp,p) = " ++ show (dp,p)
  printStringAtLsDelta [] dp str

-- ---------------------------------------------------------------------

markLocatedAA :: ApiAnn' a -> (a -> AddApiAnn) -> EPP ()
markLocatedAA ApiAnnNotUsed  _  = return ()
markLocatedAA (ApiAnn _ a _) f = mark [f a] kw
  where
    AddApiAnn kw _ = f a

-- ---------------------------------------------------------------------

-- markLocatedMaybe :: a -> (a -> Maybe RealSrcSpan) -> AnnKeywordId -> EPP ()
-- markLocatedMaybe a

-- ---------------------------------------------------------------------

markAnnOpen :: Maybe RealSrcSpan -> SourceText -> String -> EPP ()
markAnnOpen ms NoSourceText txt   = printStringAtMkw ms txt
markAnnOpen ms (SourceText txt) _ = printStringAtMkw ms txt

-- ---------------------------------------------------------------------

markAnnKw :: ApiAnn' a -> (a -> RealSrcSpan) -> AnnKeywordId -> EPP ()
markAnnKw ApiAnnNotUsed  _ _  = return ()
markAnnKw (ApiAnn _ a _) f kw = markKw' kw (f a)

markALocatedA :: ApiAnn' AnnListItem -> AnnKeywordId -> EPP ()
markALocatedA ApiAnnNotUsed  _  = return ()
markALocatedA (ApiAnn _ a _) kw = mark (lann_trailing a) kw

-- markALocatedN :: ApiAnn' NameAnn -> AnnKeywordId -> EPP ()
-- markALocatedN ApiAnnNotUsed  _  = return ()
-- markALocatedN (ApiAnn _ a _) kw = mark (nann_trailing a) kw

markApiAnn :: ApiAnn -> AnnKeywordId -> EPP ()
markApiAnn ApiAnnNotUsed _ = return ()
markApiAnn (ApiAnn _ a _) kw = mark a kw


mark :: [AddApiAnn] -> AnnKeywordId -> EPP ()
mark anns kw = do
  case find (\(AddApiAnn k _) -> k == kw) anns of
    Nothing -> return ()
    Just aa -> markKw aa

markKw :: AddApiAnn -> EPP ()
markKw (AddApiAnn kw ss) = markKw' kw ss

-- | This should be the main driver of the process, managing comments
markKw' :: AnnKeywordId -> RealSrcSpan -> EPP ()
markKw' kw ss = do
  p' <- getPos
  printComments ss
  dp <- nextDP ss
  p <- getPos
  debugM $ "markKw: (dp,p,p') = " ++ show (dp,p,p')
  printStringAtLsDelta [] dp (keywordToString (G kw))

-- ---------------------------------------------------------------------

-- printTrailingComments :: EPP ()
-- printTrailingComments = do
--   cs <- getUnallocatedComments
--   mapM_ printOneComment cs

-- ---------------------------------------------------------------------

printComments :: RealSrcSpan -> EPP ()
printComments ss = do
  cs <- commentAllocation ss
  debugM $ "printComments: (ss,comment locations): " ++ showGhc (ss,map commentIdentifier cs)
  mapM_ printOneComment cs

-- ---------------------------------------------------------------------

printOneComment :: Comment -> EPP ()
printOneComment c@(Comment _str loc _mo) = do
  p <- getPos
  let dp = ss2delta p loc
  printQueuedComment c dp

-- ---------------------------------------------------------------------

commentAllocation :: RealSrcSpan -> EPP [Comment]
commentAllocation ss = do
  cs <- getUnallocatedComments
  let (earlier,later) = partition (\(Comment _str loc _mo) -> loc <= ss) cs
  putUnallocatedComments later
  return earlier

-- ---------------------------------------------------------------------

-- commentAllocation :: (Comment -> Bool)
--                   -> EPP a
-- commentAllocation p = do
--   cs <- getUnallocatedComments
--   let (allocated,cs') = allocateComments p cs
--   putUnallocatedComments cs'
--   mapM makeDeltaComment (sortBy (comparing commentIdentifier) allocated)

-- makeDeltaComment :: Comment -> EPP (Comment, DeltaPos)
-- makeDeltaComment c = do
--   let pa = commentIdentifier c
--   pe <- getPriorEnd
--   let p = ss2delta pe pa
--   p' <- adjustDeltaForOffsetM p
--   setPriorEnd (ss2posEnd pa)
--   return (c, p')


-- ---------------------------------------------------------------------

nextDP :: RealSrcSpan -> EPP DeltaPos
nextDP ss = do
  p <- getPos
  return $ pos2delta p (ss2pos ss)

-- ---------------------------------------------------------------------

markListWithLayout :: ExactPrint (LocatedA ast) => [LocatedA ast] -> EPP ()
markListWithLayout ls =
  setLayout $ markListA ls

-- ---------------------------------------------------------------------

markList :: ExactPrint ast => [Located ast] -> EPP ()
markList ls =
  -- setContext (Set.singleton NoPrecedingSpace)
  --  $ markListWithContexts' listContexts' ls
  mapM_ markAnnotated ls

markListA :: ExactPrint (LocatedA ast) => [LocatedA ast] -> EPP ()
markListA ls =
  -- setContext (Set.singleton NoPrecedingSpace)
  --  $ markListWithContexts' listContexts' ls
  mapM_ markAnnotated ls

-- ---------------------------------------------------------------------


------------------------------
{-
instance Annotate (GHC.HsModule GHC.GhcPs) where
  markAST _ (GHC.HsModule mmn mexp imps decs mdepr _haddock) = do

    case mmn of
      Nothing -> return ()
      Just (GHC.L ln mn) -> do
        mark GHC.AnnModule
        markExternal ln GHC.AnnVal (GHC.moduleNameString mn)

        forM_ mdepr markLocated
        forM_ mexp markLocated

        mark GHC.AnnWhere

    markOptional GHC.AnnOpenC -- Possible '{'
    markManyOptional GHC.AnnSemi -- possible leading semis
    setContextLevel (Set.singleton TopLevel) 2 $ markListWithLayout imps

    setContextLevel (Set.singleton TopLevel) 2 $ markListWithLayout decs

    markOptional GHC.AnnCloseC -- Possible '}'

    markEOF
-}
------------------------------

    -- exact (HsModule anns Nothing _ imports decls _ mbDoc)
    --   = pp_mb mbDoc $$ pp_nonnull imports
    --                 $$ pp_nonnull decls

    -- exact (HsModule _ (Just name) exports imports decls deprec mbDoc)
    --   = vcat [
    --         pp_mb mbDoc,
    --         case exports of
    --           Nothing -> pp_header (text "where")
    --           Just es -> vcat [
    --                        pp_header lparen,
    --                        nest 8 (fsep (punctuate comma (map exact (unLoc es)))),
    --                        nest 4 (text ") where")
    --                       ],
    --         pp_nonnull imports,
    --         pp_nonnull decls
    --       ]
    --   where
    --     pp_header rest = case deprec of
    --        Nothing -> pp_modname <+> rest
    --        Just d -> vcat [ pp_modname, exact d, rest ]

    --     pp_modname = text "module" <+> exact name

-- ---------------------------------------------------------------------

-- pp_mb :: ExactPrint t => Maybe t -> SDoc
-- pp_mb (Just x) = ppr x
-- pp_mb Nothing  = empty

-- pp_nonnull :: ExactPrint t => [t] -> SDoc
-- pp_nonnull [] = empty
-- pp_nonnull xs = vcat (map ppr xs)

-- ---------------------------------------------------------------------

instance ExactPrint ModuleName where
  getApiAnnotation _ = NoEntryVal
  exact = withPpr

-- ---------------------------------------------------------------------

-- instance ExactPrint (LocatedA WarningTxt) where
--   exact (L _ a) = withPpr a -- TODO:AZ: use annotations

-- ---------------------------------------------------------------------

-- instance ExactPrint (LIE GhcPs) where
--   exact = withPpr -- TODO:AZ use annotations

-- ---------------------------------------------------------------------

-- instance ExactPrint (LHsDecl GhcPs) where
--   exact = withPpr -- TODO:AZ use annotations

-- ---------------------------------------------------------------------

instance ExactPrint (LImportDecl GhcPs) where
  getApiAnnotation (L ann _) = fromAnn ann
  exact (L _ a) = exact a
    -- Used to print the annotations related to being in a
    -- list. Perhaps rely on the generic LocatedA one?

instance ExactPrint (ImportDecl GhcPs) where
  getApiAnnotation idecl = fromAnn (ideclExt idecl)
  exact x@(ImportDecl ApiAnnNotUsed _ _ _ _ _ _ _ _ _) = withPpr x
  exact (ImportDecl ann@(ApiAnn _ an _) msrc (L lm modname) mpkg _src safeflag qualFlag _impl mAs hiding) = do

    markAnnKw ann importDeclAnnImport AnnImport

    -- "{-# SOURCE" and "#-}"
    case msrc of
      SourceText _txt -> do
        debugM $ "ImportDecl sourcetext"
        let mo = fmap fst $ importDeclAnnPragma an
        let mc = fmap snd $ importDeclAnnPragma an
        markAnnOpen mo msrc "{-# SOURCE"
        printStringAtMkw mc "#-}"
      NoSourceText -> return ()
 --   when safeflag (mark GHC.AnnSafe)
    case qualFlag of
      QualifiedPre  -- 'qualified' appears in prepositive position.
        -> printStringAtMkw (importDeclAnnQualified an) "qualified"
      _ -> return ()
 --   case mpkg of
 --    Just (GHC.StringLiteral (GHC.SourceText srcPkg) _) ->
 --      markWithString GHC.AnnPackageName srcPkg
 --    _ -> return ()

    printStringAtKw' (realSrcSpan lm) (moduleNameString modname)

    case qualFlag of
      QualifiedPost  -- 'qualified' appears in postpositive position.
        -> printStringAtMkw (importDeclAnnQualified an) "qualified"
      _ -> return ()

    case mAs of
      Nothing -> return ()
      Just (L l mn) -> do
        printStringAtMkw (importDeclAnnAs an) "as"
        printStringAtKw' (realSrcSpan l) (moduleNameString mn)

    case hiding of
      Nothing -> return ()
      Just (_isHiding,lie) -> exact lie
 --   markTrailingSemi


-- ---------------------------------------------------------------------

instance ExactPrint HsDocString where
  getApiAnnotation _ = NoEntryVal
  exact = withPpr -- TODO:AZ use annotations

-- ---------------------------------------------------------------------

-- instance (ExactPrint a) => ExactPrint (LocatedA a) where
--   exact (L _ a) = exact a -- TODO:AZ: use annotations

-- ---------------------------------------------------------------------

instance ExactPrint (LHsDecl GhcPs) where
  -- getApiAnnotation (L _ (TyClD      _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (InstD      _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (DerivD     _ d)) = getApiAnnotation d
  getApiAnnotation (L _ (ValD       _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (SigD       _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (KindSigD   _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (DefD       _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (ForD       _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (WarningD   _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (AnnD       _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (RuleD      _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (SpliceD    _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (DocD       _ d)) = getApiAnnotation d
  -- getApiAnnotation (L _ (RoleAnnotD _ d)) = getApiAnnotation d

  exact (L _ (ValD _ d)) = exact d -- TODO:AZ use annotations
  exact d = withPpr d -- TODO:AZ use annotations


-- ---------------------------------------------------------------------

instance ExactPrint (HsBind GhcPs) where
  getApiAnnotation FunBind{} = NoEntryVal

  exact (FunBind _ _ (MG _ (L _ matches) _) _) = do
    markList matches
  exact b = withPpr b


-- ---------------------------------------------------------------------

instance ExactPrint (Match GhcPs (LHsExpr GhcPs)) where
  getApiAnnotation (Match ann _ _ _) = fromAnn ann

  exact match@(Match ApiAnnNotUsed _ _ _) = withPpr match
  exact (Match anns@(ApiAnn _ ann cs) mctxt pats (GRHSs _ grhs (L _ lb))) = do
  -- markAST _ (GHC.Match _ mln pats (GHC.GRHSs _ grhs (GHC.L _ lb))) = do
    debugM $ "exact Match entered"
    let
      get_infix (FunRhs _ f _) = f
      get_infix _              = Prefix

      isFunBind FunRhs{} = True
      isFunBind _        = False

    case mctxt of
      FunRhs fun _fixity _strictness -> do
        debugM $ "exact Match FunRhs:" ++ showGhc fun
        -- exact fun
        markAnnotated fun
      _ -> withPpr mctxt

    case grhs of
      (GHC.L _ (GHC.GRHS _ [] _):_) -> when (isFunBind mctxt) $ markApiAnn anns AnnEqual -- empty guards
      _ -> return ()
    case mctxt of
      LambdaExpr -> markApiAnn anns AnnRarrow -- For HsLam
      _ -> return ()

    mapM_ markAnnotated grhs

-- ---------------------------------------------------------------------

instance ExactPrint (GRHS GhcPs (LHsExpr GhcPs)) where
  getApiAnnotation (GRHS ann _ _) = fromAnn ann

  exact (GRHS anns guards expr) = do
    debugM $ "GRHS: anns=" ++ showGhc anns
    case guards of
      [] -> return ()
      (_:_) -> do
        markApiAnn anns AnnVbar
        -- unsetContext Intercalate $ setContext (Set.fromList [LeftMost,PrefixOp])
        --   $ markListIntercalate guards
        -- ifInContext (Set.fromList [CaseAlt])
        --   (return ())
        --   (mark GHC.AnnEqual)
        markListA guards
        markApiAnn anns AnnEqual

    -- markOptional GHC.AnnEqual -- For apply-refact Structure8.hs test

    -- inContext (Set.fromList [CaseAlt]) $ mark GHC.AnnRarrow -- For HsLam
    -- setContextLevel (Set.fromList [LeftMost,PrefixOp]) 2 $ markLocated expr

    markAnnotated expr
  -- markAST _ (GHC.XGRHS x) = error $ "got XGRHS for:" ++ showGhc x

-- ---------------------------------------------------------------------

instance ExactPrint (LocatedA (StmtLR GhcPs GhcPs (LHsExpr GhcPs))) where
  getApiAnnotation = undefined
  exact = withPpr -- AZ TODO

-- ---------------------------------------------------------------------

instance ExactPrint (LHsExpr GhcPs) where
  getApiAnnotation = entryFromLocatedA
  exact (L _ a) = do
    debugM $ "exact:LHsExpr:" ++ showGhc a
    markAnnotated a

instance ExactPrint (HsExpr GhcPs) where
  getApiAnnotation (HsVar{})                    = NoEntryVal
  getApiAnnotation (HsUnboundVar ann _)         = fromAnn ann
  getApiAnnotation (HsConLikeOut{})             = NoEntryVal
  getApiAnnotation (HsRecFld{})                 = NoEntryVal
  getApiAnnotation (HsOverLabel ann _ _)        = fromAnn ann
  getApiAnnotation (HsIPVar ann _)              = fromAnn ann
  getApiAnnotation (HsOverLit ann _)            = fromAnn ann
  getApiAnnotation (HsLit ann _)                = fromAnn ann
  getApiAnnotation (HsLam ann _)                = fromAnn ann
  getApiAnnotation (HsLamCase ann _)            = fromAnn ann
  getApiAnnotation (HsApp ann _ _)              = fromAnn ann
  getApiAnnotation (HsAppType ann _ _)          = fromAnn ann
  getApiAnnotation (OpApp ann _ _ _)            = fromAnn ann
  getApiAnnotation (NegApp ann _ _)             = fromAnn ann
  getApiAnnotation (HsPar ann _)                = fromAnn ann
  getApiAnnotation (SectionL ann _ _)           = fromAnn ann
  getApiAnnotation (SectionR ann _ _)           = fromAnn ann
  getApiAnnotation (ExplicitTuple ann _ _)      = fromAnn ann
  getApiAnnotation (ExplicitSum ann _ _ _)      = fromAnn ann
  getApiAnnotation (HsCase (ApiAnn a _ cs) _ _) = Entry a cs
  getApiAnnotation (HsCase ApiAnnNotUsed   _ _) = NoEntryVal
  getApiAnnotation (HsIf (ann,_) _ _ _ _)       = fromAnn ann
  getApiAnnotation (HsMultiIf ann _)            = fromAnn ann
  getApiAnnotation (HsLet ann _ _)              = fromAnn ann
  getApiAnnotation (HsDo ann _ _)               = fromAnn ann
  getApiAnnotation (ExplicitList ann _ _)       = fromAnn ann
  getApiAnnotation (RecordCon ann _ _)          = fromAnn ann
  getApiAnnotation (RecordUpd ann _ _)          = fromAnn ann
  getApiAnnotation (ExprWithTySig ann _ _)      = fromAnn ann
  getApiAnnotation (ArithSeq ann _ _)           = fromAnn ann
  getApiAnnotation (HsBracket ann _)            = fromAnn ann
  getApiAnnotation (HsRnBracketOut{})           = NoEntryVal
  getApiAnnotation (HsTcBracketOut{})           = NoEntryVal
  getApiAnnotation (HsSpliceE ann _)            = fromAnn ann
  getApiAnnotation (HsProc ann _ _)             = fromAnn ann
  getApiAnnotation (HsStatic{})                 = NoEntryVal
  getApiAnnotation (HsTick {})                  = NoEntryVal
  getApiAnnotation (HsBinTick {})               = NoEntryVal
  getApiAnnotation (HsPragE{})                  = NoEntryVal


  exact (HsVar _ n) = markAnnotated n
  -- exact x@(HsUnboundVar ann _)         = withPpr x
  -- exact x@(HsConLikeOut{})             = withPpr x
  -- exact x@(HsRecFld{})                 = withPpr x
  -- exact x@(HsOverLabel ann _ _)        = withPpr x
  -- exact x@(HsIPVar ann _)              = withPpr x
  exact x@(HsOverLit ann ol) = do
    let str = case ol_val ol of
                HsIntegral   (IL src _ _) -> src
                HsFractional (FL src _ _) -> src
                HsIsString src _          -> src
    -- markExternalSourceText l str ""
    case str of
      SourceText s -> printString False s
      NoSourceText -> withPpr x

  exact (HsLit ann lit) = withPpr lit
  -- exact x@(HsLam ann _)                = withPpr x
  -- exact x@(HsLamCase ann _)            = withPpr x
  exact (HsApp ann e1 e2) = do
    p <- getPos
    debugM $ "HsApp entered. p=" ++ show p
    markAnnotated e1
    markAnnotated e2
  -- exact x@(HsAppType ann _ _)          = withPpr x
  -- exact x@(OpApp ann _ _ _)            = withPpr x
  -- exact x@(NegApp ann _ _)             = withPpr x
  -- exact x@(HsPar ann _)                = withPpr x
  -- exact x@(SectionL ann _ _)           = withPpr x
  -- exact x@(SectionR ann _ _)           = withPpr x
  -- exact x@(ExplicitTuple ann _ _)      = withPpr x
  -- exact x@(ExplicitSum ann _ _ _)      = withPpr x
  -- exact x@(HsCase (ApiAnn a _ cs) _ _) = withPpr x
  -- exact x@(HsCase ApiAnnNotUsed   _ _) = withPpr x
  -- exact x@(HsIf (ann,_) _ _ _ _)       = withPpr x
  -- exact x@(HsMultiIf ann _)            = withPpr x
  -- exact x@(HsLet ann _ _)              = withPpr x
  -- exact x@(HsDo ann _ _)               = withPpr x
  -- exact x@(ExplicitList ann _ _)       = withPpr x
  -- exact x@(RecordCon ann _ _)          = withPpr x
  -- exact x@(RecordUpd ann _ _)          = withPpr x
  -- exact x@(ExprWithTySig ann _ _)      = withPpr x
  -- exact x@(ArithSeq ann _ _)           = withPpr x
  -- exact x@(HsBracket ann _)            = withPpr x
  -- exact x@(HsRnBracketOut{})           = withPpr x
  -- exact x@(HsTcBracketOut{})           = withPpr x
  -- exact x@(HsSpliceE ann _)            = withPpr x
  -- exact x@(HsProc ann _ _)             = withPpr x
  -- exact x@(HsStatic{})                 = withPpr x
  -- exact x@(HsTick {})                  = withPpr x
  -- exact x@(HsBinTick {})               = withPpr x
  -- exact x@(HsPragE{})                  = withPpr x
  exact x = error $ "exact HsExpr for:" ++ showAst x

-- ---------------------------------------------------------------------

instance ExactPrint (LocatedN RdrName) where
  getApiAnnotation (L sann _) = fromAnn sann

  exact (L (SrcSpanAnn ApiAnnNotUsed _) n) = do
    printString False (showGhc n)
  exact (L (SrcSpanAnn (ApiAnn _anchor ann _cs) _) n) = do
    case ann of
      NameAnn a o l c t -> do
        markName a o (Just (l,n)) c
        markTrailing t
      NameAnnCommas a o cs c t -> do
        markName a o Nothing c
        markTrailing t
      NameAnnOnly a o c t -> do
        markName a o Nothing c
        markTrailing t
      NameAnnRArrow nl t -> do
        markKw (AddApiAnn AnnRarrow nl)
        markTrailing t
      NameAnnTrailing t -> do
        markTrailing t

markName :: NameAdornment
         -> RealSrcSpan -> Maybe (RealSrcSpan,RdrName) -> RealSrcSpan -> EPP ()
markName adorn open mname close = do
  let (kwo,kwc) = adornments adorn
  markKw (AddApiAnn kwo open)
  case mname of
    Nothing -> return ()
    Just (name, a) -> printStringAtKw' name (showGhc a)
  markKw (AddApiAnn kwc close)
  where
    adornments :: NameAdornment -> (AnnKeywordId, AnnKeywordId)
    adornments NameParens     = (AnnOpenP, AnnCloseP)
    adornments NameParensHash = (AnnOpenPH, AnnClosePH)
    adornments NameBackquotes = (AnnBackquote, AnnBackquote)
    adornments NameSquare     = (AnnOpenS, AnnCloseS)

markTrailing :: [AddApiAnn] -> EPP ()
markTrailing ts = do
  mapM_ markKw ts

-- ---------------------------------------------------------------------

instance ExactPrint (LocatedL [LIE GhcPs]) where
  getApiAnnotation (L (SrcSpanAnn ann _) _) = fromAnn ann
  exact (L (SrcSpanAnn ann _) ies) = do
    -- markLocatedL ann AnnOpenP
    markLocatedAA ann al_open
    mapM_ markAnnotated ies
    markLocatedAA ann al_close
    -- markLocatedL ann AnnCloseP

-- ---------------------------------------------------------------------

instance ExactPrint (LIE GhcPs) where
  getApiAnnotation _ = NoEntryVal
  exact (L (SrcSpanAnn ann _) a) = do
    markAnnotated a
    markALocatedA ann AnnComma

instance ExactPrint (IE GhcPs) where
  getApiAnnotation (IEVar anns _)             = fromAnn anns
  getApiAnnotation (IEThingAbs anns _)        = fromAnn anns
  getApiAnnotation (IEThingAll anns _)        = fromAnn anns
  getApiAnnotation (IEThingWith anns _ _ _ _) = fromAnn anns
  getApiAnnotation (IEModuleContents anns _)  = fromAnn anns
  getApiAnnotation (IEGroup _ _ _)            = NoEntryVal
  getApiAnnotation (IEDoc _ _)                = NoEntryVal
  getApiAnnotation (IEDocNamed _ _)           = NoEntryVal

  exact = withPpr

-- ---------------------------------------------------------------------

entryFromLocatedA :: LocatedA a -> Entry
entryFromLocatedA (L (SrcSpanAnn ann _) _) = fromAnn ann


-- =====================================================================
-- Utility stuff
-- annNone :: Annotation
-- annNone = Ann (DP (0,0)) [] [] [] Nothing Nothing

-- -- ---------------------------------------------------------------------
-- -- | Calculates the distance from the start of a string to the end of
-- -- a string.
-- dpFromString ::  String -> DeltaPos
-- dpFromString xs = dpFromString' xs 0 0
--   where
--     dpFromString' "" line col = DP (line, col)
--     dpFromString' ('\n': cs) line _   = dpFromString' cs (line + 1) 0
--     dpFromString' (_:cs)     line col = dpFromString' cs line       (col + 1)

-- ---------------------------------------------------------------------
-- ---------------------------------------------------------------------

-- ---------------------------------------------------------------------

-- | Put the provided context elements into the existing set with fresh level
-- counts
-- setAcs :: Set.Set AstContext -> AstContextSet -> AstContextSet
-- setAcs ctxt acs = setAcsWithLevel ctxt 3 acs

-- -- | Put the provided context elements into the existing set with given level
-- -- counts
-- -- setAcsWithLevel :: Set.Set AstContext -> Int -> AstContextSet -> AstContextSet
-- -- setAcsWithLevel ctxt level (ACS a) = ACS a'
-- --   where
-- --     upd s (k,v) = Map.insert k v s
-- --     a' = foldl' upd a $ zip (Set.toList ctxt) (repeat level)
-- setAcsWithLevel :: (Ord a) => Set.Set a -> Int -> ACS' a -> ACS' a
-- setAcsWithLevel ctxt level (ACS a) = ACS a'
--   where
--     upd s (k,v) = Map.insert k v s
--     a' = foldl' upd a $ zip (Set.toList ctxt) (repeat level)

-- ---------------------------------------------------------------------
-- | Remove the provided context element from the existing set
-- unsetAcs :: AstContext -> AstContextSet -> AstContextSet
-- unsetAcs :: (Ord a) => a -> ACS' a -> ACS' a
-- unsetAcs ctxt (ACS a) = ACS $ Map.delete ctxt a

-- ---------------------------------------------------------------------

-- | Are any of the contexts currently active?
-- inAcs :: Set.Set AstContext -> AstContextSet -> Bool
-- inAcs :: (Ord a) => Set.Set a -> ACS' a -> Bool
-- inAcs ctxt (ACS a) = not $ Set.null $ Set.intersection ctxt (Set.fromList $ Map.keys a)

-- -- | propagate the ACS down a level, dropping all values which hit zero
-- -- pushAcs :: AstContextSet -> AstContextSet
-- pushAcs :: ACS' a -> ACS' a
-- pushAcs (ACS a) = ACS $ Map.mapMaybe f a
--   where
--     f n
--       | n <= 1    = Nothing
--       | otherwise = Just (n - 1)

-- |Sometimes we have to pass the context down unchanged. Bump each count up by
-- one so that it is unchanged after a @pushAcs@ call.
-- bumpAcs :: AstContextSet -> AstContextSet
-- bumpAcs :: ACS' a -> ACS' a
-- bumpAcs (ACS a) = ACS $ Map.mapMaybe f a
--   where
--     f n = Just (n + 1)


-- ---------------------------------------------------------------------
-- ---------------------------------------------------------------------

printStringAtMaybeAnn :: (Monad m, Monoid w) => KeywordId -> Maybe String -> EP w m ()
printStringAtMaybeAnn an mstr = printStringAtMaybeAnnThen an mstr (return ())

-- printStringAtMaybeAnnAll :: (Monad m, Monoid w) => KeywordId -> Maybe String -> EP w m ()
-- printStringAtMaybeAnnAll an mstr = go
--   where
--     go = printStringAtMaybeAnnThen an mstr go

printStringAtMaybeAnnThen :: (Monad m, Monoid w)
                          => KeywordId -> Maybe String -> EP w m () -> EP w m ()
printStringAtMaybeAnnThen an mstr next = do
  let str = fromMaybe (keywordToString an) mstr
  annFinal <- getAnnFinal an
  case (annFinal, an) of
    -- Could be unicode syntax
    -- TODO: This is a bit fishy, refactor
    (Nothing, G kw') -> do
      let kw = unicodeAnn kw'
      let str' = fromMaybe (keywordToString (G kw)) mstr
      res <- getAnnFinal (G kw)
      return () `debug` ("printStringAtMaybeAnn:missed:Unicode:(an,res)" ++ show (an,res))
      unless (null res) $ do
        forM_
          res
          (\(comments, ma) -> printStringAtLsDelta comments ma str')
        next
    (Just (comments, ma),_) -> printStringAtLsDelta comments ma str >> next
    (Nothing, _) -> return () `debug` ("printStringAtMaybeAnn:missed:(an)" ++ show an)
                    -- Note: do not call next, nothing to chain
    -- ++AZ++: Enabling the following line causes a very weird error associated with AnnPackageName. I suspect it is because it is forcing the evaluation of a non-existent an or str
    -- `debug` ("printStringAtMaybeAnn:(an,ma,str)=" ++ show (an,ma,str))

-- ---------------------------------------------------------------------

-- |This should be the final point where things are mode concrete,
-- before output. Hence the point where comments can be inserted
printStringAtLsDelta :: (Monad m, Monoid w) => [(Comment, DeltaPos)] -> DeltaPos -> String -> EP w m ()
printStringAtLsDelta cs cl s = do
  p <- getPos
  colOffset <- getLayoutOffset
  if isGoodDeltaWithOffset cl colOffset
    then do
      mapM_ (uncurry printQueuedComment) cs
      printStringAt (undelta p cl colOffset) s
        `debug` ("printStringAtLsDelta:(pos,s):" ++ show (undelta p cl colOffset,s))
    else return () `debug` ("printStringAtLsDelta:bad delta for (mc,s):" ++ show (cl,s))

-- ---------------------------------------------------------------------

-- |destructive get, hence use an annotation once only
getAnnFinal :: (Monad m, Monoid w)
  => KeywordId -> EP w m (Maybe ([(Comment, DeltaPos)], DeltaPos))
getAnnFinal kw = do
  kd <- gets epAnnKds
  case kd of
    []    -> return Nothing -- Should never be triggered
    (k:kds) -> do
      let (res, kd') = destructiveGetFirst kw ([],k)
      modify (\s -> s { epAnnKds = kd' : kds })
      return res

-- | Get and remove the first item in the (k,v) list for which the k matches.
-- Return the value, together with any comments skipped over to get there.
destructiveGetFirst :: KeywordId
                    -> ([(KeywordId, v)],[(KeywordId,v)])
                    -> (Maybe ([(Comment, v)], v),[(KeywordId,v)])
destructiveGetFirst _key (acc,[]) = (Nothing, acc)
destructiveGetFirst  key (acc, (k,v):kvs )
  | k == key = (Just (skippedComments, v), others ++ kvs)
  | otherwise = destructiveGetFirst key (acc ++ [(k,v)], kvs)
  where
    (skippedComments, others) = foldr comments ([], []) acc
    comments (AnnComment comment' , dp ) (cs, kws) = ((comment', dp) : cs, kws)
    comments kw (cs, kws)                          = (cs, kw : kws)



isGoodDeltaWithOffset :: DeltaPos -> LayoutStartCol -> Bool
isGoodDeltaWithOffset dp colOffset = isGoodDelta (DP (undelta (0,0) dp colOffset))

printQueuedComment :: (Monad m, Monoid w) => Comment -> DeltaPos -> EP w m ()
printQueuedComment Comment{commentContents} dp = do
  p <- getPos
  colOffset <- getLayoutOffset
  let (dr,dc) = undelta (0,0) dp colOffset
  debugM $ "printQueuedComment: (p,dp,colOffset,undelta)=" ++ show (p,dp,colOffset,undelta p dp colOffset)
  -- do not lose comments against the left margin
  when (isGoodDelta (DP (dr,max 0 dc))) $
    printCommentAt (undelta p dp colOffset) commentContents


-- ---------------------------------------------------------------------

-- withContext :: (Monad m, Monoid w)
--             => [(KeywordId, DeltaPos)]
--             -> Annotation
--             -> EP w m a -> EP w m a
-- withContext kds an x = withKds kds (withOffset an x)

-- ---------------------------------------------------------------------
--
-- | Given an annotation associated with a specific SrcSpan,
-- determines a new offset relative to the previous offset
--
withOffset :: (Monad m, Monoid w) => Annotation -> (EP w m a -> EP w m a)
withOffset a =
  local (\s -> s { epAnn = a, epContext = pushAcs (epContext s) })


-- ---------------------------------------------------------------------
--
-- Necessary as there are destructive gets of Kds across scopes
-- withKds :: (Monad m, Monoid w) => [(KeywordId, DeltaPos)] -> EP w m a -> EP w m a
-- withKds kd action = do
--   modify (\s -> s { epAnnKds = kd : epAnnKds s })
--   r <- action
--   modify (\s -> s { epAnnKds = tail (epAnnKds s) })
--   return r

------------------------------------------------------------------------

setLayout :: (Monad m, Monoid w) => EP w m () -> EP w m ()
setLayout k = do
  oldLHS <- gets epLHS
  modify (\a -> a { epMarkLayout = True } )
  let reset = modify (\a -> a { epMarkLayout = False
                              , epLHS = oldLHS } )
  k <* reset

getPos :: (Monad m, Monoid w) => EP w m Pos
getPos = gets epPos

setPos :: (Monad m, Monoid w) => Pos -> EP w m ()
setPos l = modify (\s -> s {epPos = l})

getUnallocatedComments :: (Monad m, Monoid w) => EP w m [Comment]
getUnallocatedComments = gets epComments

putUnallocatedComments :: (Monad m, Monoid w) => [Comment] -> EP w m ()
putUnallocatedComments cs = modify (\s -> s { epComments = cs } )

-- |Get the current column offset
getLayoutOffset :: (Monad m, Monoid w) => EP w m LayoutStartCol
getLayoutOffset = gets epLHS

getEofPos :: (Monad m, Monoid w) => EP w m RealSrcSpan
getEofPos = do
  as <- gets epApiAnns
  case apiAnnEofPos as of
    Nothing -> return placeholderRealSpan
    Just ss -> return ss

-- ---------------------------------------------------------------------
-------------------------------------------------------------------------
-- |First move to the given location, then call exactP
-- exactPC :: (Data ast, Monad m, Monoid w) => GHC.Located ast -> EP w m a -> EP w m a
-- exactPC :: (Data ast, Data (GHC.SrcSpanLess ast), GHC.HasSrcSpan ast, Monad m, Monoid w)
-- exactPC :: (Data ast, Monad m, Monoid w) => GHC.Located ast -> EP w m a -> EP w m a
-- exactPC ast action =
--     do
--       return () `debug` ("exactPC entered for:" ++ show (mkAnnKey ast))
--       ma <- getAndRemoveAnnotation ast
--       let an@Ann{ annEntryDelta=edp
--                 , annPriorComments=comments
--                 , annFollowingComments=fcomments
--                 , annsDP=kds
--                 } = fromMaybe annNone ma
--       PrintOptions{epAstPrint} <- ask
--       r <- withContext kds an
--        (mapM_ (uncurry printQueuedComment) comments
--        >> advance edp
--        >> censorM (epAstPrint ast) action
--        <* mapM_ (uncurry printQueuedComment) fcomments)
--       return r `debug` ("leaving exactPCfor:" ++ show (mkAnnKey ast))

-- censorM :: (Monoid w, Monad m) => (w -> m w) -> EP w m a -> EP w m a
-- censorM f m = passM (liftM (\x -> (x,f)) m)

-- passM :: (Monad m) => EP w m (a, w -> m w) -> EP w m a
-- passM m = RWST $ \r s -> do
--       ~((a, f),s', EPWriter w) <- runRWST m r s
--       w' <- f w
--       return (a, s', EPWriter w')

advance :: (Monad m, Monoid w) => DeltaPos -> EP w m ()
advance cl = do
  p <- getPos
  colOffset <- getLayoutOffset
  debugM $ "advance:(p,colOffset,ws)=" ++ show (p,colOffset,undelta p cl colOffset)
  printWhitespace (undelta p cl colOffset)

-- getAndRemoveAnnotation :: (Monad m, Monoid w, Data a) => GHC.Located a -> EP w m (Maybe Annotation)
-- getAndRemoveAnnotation a = gets (getAnnotationEP a . epAnns)

-- ---------------------------------------------------------------------

-- adjustDeltaForOffsetM :: DeltaPos -> EPP DeltaPos
-- adjustDeltaForOffsetM dp = do
--   colOffset <- gets epLHS
--   return (adjustDeltaForOffset colOffset dp)

adjustDeltaForOffset :: LayoutStartCol -> DeltaPos -> DeltaPos
adjustDeltaForOffset _colOffset              dp@(DP (0,_)) = dp -- same line
adjustDeltaForOffset (LayoutStartCol colOffset) (DP (l,c)) = DP (l,c - colOffset)

-- ---------------------------------------------------------------------
-- Printing functions




printString :: (Monad m, Monoid w) => Bool -> String -> EP w m ()
printString layout str = do
  EPState{epPos = (_,c), epMarkLayout} <- get
  PrintOptions{epTokenPrint, epWhitespacePrint} <- ask
  when (epMarkLayout && layout) $
    modify (\s -> s { epLHS = LayoutStartCol c, epMarkLayout = False } )

  -- Advance position, taking care of any newlines in the string
  let strDP@(DP (cr,_cc)) = dpFromString str
  p <- getPos
  colOffset <- getLayoutOffset
  if cr == 0
    then setPos (undelta p strDP colOffset)
    else setPos (undelta p strDP 1)

  -- Debug stuff
  pp <- getPos
  debugM $ "printString: (p,pp,str)" ++ show (p,pp,str)
  -- Debug end

  --
  if not layout && c == 0
    then lift (epWhitespacePrint str) >>= \s -> tell EPWriter { output = s}
    else lift (epTokenPrint      str) >>= \s -> tell EPWriter { output = s}


newLine :: (Monad m, Monoid w) => EP w m ()
newLine = do
    (l,_) <- getPos
    printString False "\n"
    setPos (l+1,1)

padUntil :: (Monad m, Monoid w) => Pos -> EP w m ()
padUntil (l,c) = do
    (l1,c1) <- getPos
    if | l1 == l && c1 <= c -> printString False $ replicate (c - c1) ' '
       | l1 < l             -> newLine >> padUntil (l,c)
       | otherwise          -> return ()

printWhitespace :: (Monad m, Monoid w) => Pos -> EP w m ()
printWhitespace = padUntil

printCommentAt :: (Monad m, Monoid w) => Pos -> String -> EP w m ()
printCommentAt p str = do
  debugM $ "printCommentAt: (pos,str)" ++ show (p,str)
  printWhitespace p >> printString False str

printStringAt :: (Monad m, Monoid w) => Pos -> String -> EP w m ()
printStringAt p str = printWhitespace p >> printString False str
