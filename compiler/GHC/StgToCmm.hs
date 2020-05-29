{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}

-----------------------------------------------------------------------------
--
-- Stg to C-- code generation
--
-- (c) The University of Glasgow 2004-2006
--
-----------------------------------------------------------------------------

module GHC.StgToCmm ( codeGen ) where

#include "HsVersions.h"

import GHC.Prelude as Prelude

import GHC.StgToCmm.Prof (initInfoTableProv, initCostCentres, ldvEnter)
import GHC.StgToCmm.Monad
import GHC.StgToCmm.Env
import GHC.StgToCmm.Bind
import GHC.StgToCmm.DataCon
import GHC.StgToCmm.Layout
import GHC.StgToCmm.Utils
import GHC.StgToCmm.Closure
import GHC.StgToCmm.Hpc
import GHC.StgToCmm.Ticky

import GHC.Cmm
import GHC.Cmm.Utils
import GHC.Cmm.CLabel

import GHC.Stg.Syntax
import GHC.Driver.Session
import GHC.Utils.Error

import GHC.Driver.Types
import GHC.Types.CostCentre
import GHC.Types.Id
import GHC.Types.Id.Info
import GHC.Types.RepType
import GHC.Core.DataCon
import GHC.Core.TyCon
import GHC.Unit.Module
import GHC.Utils.Outputable
import GHC.Data.Stream
import GHC.Types.Basic
import GHC.Types.Var.Set ( isEmptyDVarSet )
import GHC.SysTools.FileCleanup
import GHC.Types.Unique.FM

import GHC.Data.OrdList
import GHC.Cmm.Graph

import Data.IORef
import Control.Monad (forM_, when,void)
import GHC.Utils.Misc
import System.IO.Unsafe
import qualified Data.ByteString as BS
import GHC.Types.Unique.Map
import GHC.Types.SrcLoc
import Data.Maybe


codeGen :: DynFlags
        -> Module
        -> InfoTableProvMap
        -> [TyCon]
        -> CollectedCCs                -- (Local/global) cost-centres needing declaring/registering.
        -> [CgStgTopBinding]           -- Bindings to convert
        -> HpcInfo
        -> Stream IO CmmGroup ()       -- Output as a stream, so codegen can
                                       -- be interleaved with output

codeGen dflags this_mod (InfoTableProvMap (dcmap@(UniqMap denv)) _) data_tycons
        cost_centre_info stg_binds hpc_info
  = do  {     -- cg: run the code generator, and yield the resulting CmmGroup
              -- Using an IORef to store the state is a bit crude, but otherwise
              -- we would need to add a state monad layer.
        ; cgref <- liftIO $ newIORef =<< initC
        ; let cg :: FCode () -> Stream IO CmmGroup ()
              cg fcode = do
                cmm <- liftIO . withTimingSilent dflags (text "STG -> Cmm") (`seq` ()) $ do
                         st <- readIORef cgref
                         let (a,st') = runC dflags this_mod st (getCmm fcode)

                         -- NB. stub-out cgs_tops and cgs_stmts.  This fixes
                         -- a big space leak.  DO NOT REMOVE!
                         writeIORef cgref $! st'{ cgs_tops = nilOL,
                                                  cgs_stmts = mkNop }
                         return a
                yield cmm

               -- Note [codegen-split-init] the cmm_init block must come
               -- FIRST.  This is because when -split-objs is on we need to
               -- combine this block with its initialisation routines; see
               -- Note [pipeline-split-init].
        ; cg (mkModuleInit cost_centre_info this_mod hpc_info (convertDCMap this_mod dcmap))

        ; mapM_ (cg . cgTopBinding dflags) stg_binds

                -- Put datatype_stuff after code_stuff, because the
                -- datatype closure table (for enumeration types) to
                -- (say) PrelBase_True_closure, which is defined in
                -- code_stuff
        ; let do_tycon tycon = do
                -- Generate a table of static closures for an
                -- enumeration type Note that the closure pointers are
                -- tagged.
                 when (isEnumerationTyCon tycon) $ cg (cgEnumerationTyCon tycon)
                 -- Emit normal info_tables, just in case
                 mapM_ (cg . cgDataCon Nothing Nothing) (tyConDataCons tycon)
                 -- Emit special info tables for everything used in this module

        ; mapM_ do_tycon data_tycons
        ; mapM_ (\(dc, ns) -> forM_ ns $ \(k, ss) -> cg (cgDataCon (Just (this_mod, k)) ss dc)) (nonDetEltsUFM denv)
        }

---------------------------------------------------------------
--      Top-level bindings
---------------------------------------------------------------

{- 'cgTopBinding' is only used for top-level bindings, since they need
to be allocated statically (not in the heap) and need to be labelled.
No unboxed bindings can happen at top level.

In the code below, the static bindings are accumulated in the
@MkCgState@, and transferred into the ``statics'' slot by @forkStatics@.
This is so that we can write the top level processing in a compositional
style, with the increasing static environment being plumbed as a state
variable. -}

cgTopBinding :: DynFlags -> CgStgTopBinding -> FCode ()
cgTopBinding dflags (StgTopLifted (StgNonRec id rhs))
  = do  { let (info, fcode) = cgTopRhs dflags NonRecursive id rhs
        ; fcode
        ; addBindC info
        }

cgTopBinding dflags (StgTopLifted (StgRec pairs))
  = do  { let (bndrs, rhss) = unzip pairs
        ; let pairs' = zip bndrs rhss
              r = unzipWith (cgTopRhs dflags Recursive) pairs'
              (infos, fcodes) = unzip r
        ; addBindsC infos
        ; sequence_ fcodes
        }

cgTopBinding dflags (StgTopStringLit id str) = do
  let label = mkBytesLabel (idName id)
  -- emit either a CmmString literal or dump the string in a file and emit a
  -- CmmFileEmbed literal.
  -- See Note [Embedding large binary blobs] in GHC.CmmToAsm.Ppr
  let isNCG    = platformMisc_ghcWithNativeCodeGen $ platformMisc dflags
      isSmall  = fromIntegral (BS.length str) <= binBlobThreshold dflags
      asString = binBlobThreshold dflags == 0 || isSmall

      (lit,decl) = if not isNCG || asString
        then mkByteStringCLit label str
        else mkFileEmbedLit label $ unsafePerformIO $ do
               bFile <- newTempName dflags TFL_CurrentModule ".dat"
               BS.writeFile bFile str
               return bFile
  emitDecl decl
  addBindC (litIdInfo dflags id mkLFStringLit lit)


cgTopRhs :: DynFlags -> RecFlag -> Id -> CgStgRhs -> (CgIdInfo, FCode ())
        -- The Id is passed along for setting up a binding...

cgTopRhs dflags _rec bndr (StgRhsCon _cc con mn args)
  = cgTopRhsCon dflags bndr con mn (assertNonVoidStgArgs args)
      -- con args are always non-void,
      -- see Note [Post-unarisation invariants] in GHC.Stg.Unarise

cgTopRhs dflags rec bndr (StgRhsClosure fvs cc upd_flag args body)
  = ASSERT(isEmptyDVarSet fvs)    -- There should be no free variables
    cgTopRhsClosure dflags rec bndr cc upd_flag args body


---------------------------------------------------------------
--      Module initialisation code
---------------------------------------------------------------

mkModuleInit
        :: CollectedCCs         -- cost centre info
        -> Module
        -> HpcInfo
        -> [InfoTableEnt]
        -> FCode ()

mkModuleInit cost_centre_info this_mod hpc_info info_ents
  = do  { initHpc this_mod hpc_info
        ; initCostCentres cost_centre_info
        ; initInfoTableProv info_ents
        }


---------------------------------------------------------------
--      Generating static stuff for algebraic data types
---------------------------------------------------------------


cgEnumerationTyCon :: TyCon -> FCode ()
cgEnumerationTyCon tycon
  = do dflags <- getDynFlags
       emitRODataLits (mkLocalClosureTableLabel (tyConName tycon) NoCafRefs)
             [ CmmLabelOff (mkLocalClosureLabel (dataConName con) NoCafRefs)
                           (tagForCon dflags con)
             | con <- tyConDataCons tycon]


cgDataCon :: Maybe (Module, Int) -> Maybe (RealSrcSpan, String) -> DataCon -> FCode ()
-- Generate the entry code, info tables, and (for niladic constructor)
-- the static closure, for a constructor.
cgDataCon _ _ data_con | isUnboxedTupleCon data_con = return ()
cgDataCon mn ms data_con
  = do  {  pprTraceM "cgDataCon" (ppr mn <+> ppr ms <+> ppr data_con)
        ; dflags <- getDynFlags
        ; platform <- getPlatform
        ; let
            (tot_wds, --  #ptr_wds + #nonptr_wds
             ptr_wds) --  #ptr_wds
              = mkVirtConstrSizes dflags arg_reps

            nonptr_wds   = tot_wds - ptr_wds

            dyn_info_tbl =
              mkDataConInfoTable dflags data_con mn ms False ptr_wds nonptr_wds

            -- We're generating info tables, so we don't know and care about
            -- what the actual arguments are. Using () here as the place holder.
            arg_reps :: [NonVoid PrimRep]
            arg_reps = [ NonVoid rep_ty
                       | ty <- dataConRepArgTys data_con
                       , rep_ty <- typePrimRep ty
                       , not (isVoidRep rep_ty) ]
        ; emitClosureAndInfoTable dyn_info_tbl NativeDirectCall [] $
            -- NB: the closure pointer is assumed *untagged* on
            -- entry to a constructor.  If the pointer is tagged,
            -- then we should not be entering it.  This assumption
            -- is used in ldvEnter and when tagging the pointer to
            -- return it.
            -- NB 2: We don't set CC when entering data (WDP 94/06)
            do { tickyEnterDynCon
               ; ldvEnter (CmmReg nodeReg)
               ; tickyReturnOldCon (length arg_reps)
               ; void $ emitReturn [cmmOffsetB platform (CmmReg nodeReg) (tagForCon dflags data_con)]
               }
                    -- The case continuation code expects a tagged pointer
        }
