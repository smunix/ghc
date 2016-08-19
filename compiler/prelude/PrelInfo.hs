{-
(c) The GRASP/AQUA Project, Glasgow University, 1992-1998

\section[PrelInfo]{The @PrelInfo@ interface to the compiler's prelude knowledge}
-}

{-# LANGUAGE CPP #-}
module PrelInfo (
        -- * Known-key names
        isKnownKeyName,
        lookupKnownKeyName,
        knownKeyNames,

        wiredInIds, ghcPrimIds,
        primOpRules, builtinRules,

        ghcPrimExports,
        primOpId,

        -- * Random other things
        maybeCharLikeCon, maybeIntLikeCon,

        -- * Class categories
        isNumericClass, isStandardClass

    ) where

#include "HsVersions.h"

import Constants        ( mAX_TUPLE_SIZE )
import BasicTypes       ( Boxity(..) )
import ConLike          ( ConLike(..) )
import PrelNames
import PrelRules
import Avail
import PrimOp
import DataCon
import Id
import Name
import MkId
import TysPrim
import TysWiredIn
import HscTypes
import Class
import TyCon
import UniqFM
import Util
import {-# SOURCE #-} TcTypeNats ( typeNatTyCons )

import Data.Array

{-
************************************************************************
*                                                                      *
\subsection[builtinNameInfo]{Lookup built-in names}
*                                                                      *
************************************************************************

Notes about wired in things
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Wired-in things are Ids\/TyCons that are completely known to the compiler.
  They are global values in GHC, (e.g.  listTyCon :: TyCon).

* A wired in Name contains the thing itself inside the Name:
        see Name.wiredInNameTyThing_maybe
  (E.g. listTyConName contains listTyCon.

* The name cache is initialised with (the names of) all wired-in things

* The type environment itself contains no wired in things. The type
  checker sees if the Name is wired in before looking up the name in
  the type environment.

* MkIface prunes out wired-in things before putting them in an interface file.
  So interface files never contain wired-in things.
-}


knownKeyNames :: [Name]
-- This list is used to ensure that when you say "Prelude.map"
--  in your source code, or in an interface file,
-- you get a Name with the correct known key
-- (See Note [Known-key names] in PrelNames)
knownKeyNames
  = concat [ wired_tycon_kk_names funTyCon
           , concatMap wired_tycon_kk_names primTyCons

           , concatMap wired_tycon_kk_names wiredInTyCons
             -- Does not include tuples

           , concatMap wired_tycon_kk_names typeNatTyCons

           , concatMap (wired_tycon_kk_names . tupleTyCon Boxed) [1..mAX_TUPLE_SIZE]  -- Yuk
           , concatMap (wired_tycon_kk_names . tupleTyCon Unboxed) [1..mAX_TUPLE_SIZE]  -- Yuk

           , concatMap tycon_kk_names cTupleTyConNames
           , concatMap datacon_kk_names cTupleDataConNames
             -- Constraint tuples are known-key but not wired-in
             -- They can't show up in source code, but can appear
             -- in interface files

             -- Anonymous sums
           , map (tyConName . sumTyCon) [2..mAX_TUPLE_SIZE]  -- Yuk
           , [ dataConName $ sumDataCon alt arity
             | arity <- [2..mAX_TUPLE_SIZE]
             , alt <- [1..arity]
             ]

           , map idName wiredInIds
           , map (idName . primOpId) allThePrimOps
           , basicKnownKeyNames ]

  where
  -- All of the names associated with a known-key TyCon (where we only have its
  -- name, not the TyCon itself). This includes the names of the TyCon itself
  -- and its type rep binding.
  tycon_kk_names :: Name -> [Name]
  tycon_kk_names tc = [tc, mkPrelTyConRepName tc]

  -- All of the names associated with a known-key DataCon. This includes the
  -- names of the DataCon itself and its promoted type rep.
  datacon_kk_names :: Name -> [Name]
  datacon_kk_names dc =
      [ dc
      , mkPrelTyConRepName dc
      ]

  -- All of the names associated with a wired-in TyCon.
  -- This includes the TyCon itself, its DataCons and promoted TyCons.
  wired_tycon_kk_names :: TyCon -> [Name]
  wired_tycon_kk_names tc =
      tyConName tc : (rep_names tc ++ concatMap thing_kk_names (implicitTyConThings tc))

  wired_datacon_kk_names :: DataCon -> [Name]
  wired_datacon_kk_names dc
   = dataConName dc : rep_names (promoteDataCon dc)

  thing_kk_names :: TyThing -> [Name]
  thing_kk_names (ATyCon tc)                 = wired_tycon_kk_names tc
  thing_kk_names (AConLike (RealDataCon dc)) = wired_datacon_kk_names dc
  thing_kk_names thing                       = [getName thing]

  -- The TyConRepName for a known-key TyCon has a known key,
  -- but isn't itself an implicit thing.  Yurgh.
  -- NB: if any of the wired-in TyCons had record fields, the record
  --     field names would be in a similar situation.  Ditto class ops.
  --     But it happens that there aren't any
  rep_names tc = case tyConRepName_maybe tc of
                       Just n  -> [n]
                       Nothing -> []

-- | Given a 'Unique' lookup its associated 'Name' if it corresponds to a
-- known-key thing.
lookupKnownKeyName :: Unique -> Maybe Name
lookupKnownKeyName = lookupUFM knownKeysMap

-- | Is a 'Name' known-key?
isKnownKeyName :: Name -> Bool
isKnownKeyName n = elemUFM n knownKeysMap

knownKeysMap :: UniqFM Name
knownKeysMap = listToUFM [ (nameUnique n, n) | n <- knownKeyNames ]

{-
We let a lot of "non-standard" values be visible, so that we can make
sense of them in interface pragmas. It's cool, though they all have
"non-standard" names, so they won't get past the parser in user code.

************************************************************************
*                                                                      *
                PrimOpIds
*                                                                      *
************************************************************************
-}

primOpIds :: Array Int Id
-- A cache of the PrimOp Ids, indexed by PrimOp tag
primOpIds = array (1,maxPrimOpTag) [ (primOpTag op, mkPrimOpId op)
                                   | op <- allThePrimOps ]

primOpId :: PrimOp -> Id
primOpId op = primOpIds ! primOpTag op

{-
************************************************************************
*                                                                      *
\subsection{Export lists for pseudo-modules (GHC.Prim)}
*                                                                      *
************************************************************************

GHC.Prim "exports" all the primops and primitive types, some
wired-in Ids.
-}

ghcPrimExports :: [IfaceExport]
ghcPrimExports
 = map (avail . idName) ghcPrimIds ++
   map (avail . idName . primOpId) allThePrimOps ++
   [ AvailTC n [n] []
   | tc <- funTyCon : primTyCons, let n = tyConName tc  ]

{-
************************************************************************
*                                                                      *
\subsection{Built-in keys}
*                                                                      *
************************************************************************

ToDo: make it do the ``like'' part properly (as in 0.26 and before).
-}

maybeCharLikeCon, maybeIntLikeCon :: DataCon -> Bool
maybeCharLikeCon con = con `hasKey` charDataConKey
maybeIntLikeCon  con = con `hasKey` intDataConKey

{-
************************************************************************
*                                                                      *
\subsection{Class predicates}
*                                                                      *
************************************************************************
-}

isNumericClass, isStandardClass :: Class -> Bool

isNumericClass     clas = classKey clas `is_elem` numericClassKeys
isStandardClass    clas = classKey clas `is_elem` standardClassKeys

is_elem :: Eq a => a -> [a] -> Bool
is_elem = isIn "is_X_Class"
