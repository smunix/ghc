-----------------------------------------------------------------------------
-- (c) The University of Glasgow, 2006
--
-- GHC's lexer for Haskell 2010 [1].
--
-- This is a combination of an Alex-generated lexer [2] from a regex
-- definition, with some hand-coded bits. [3]
--
-- Completely accurate information about token-spans within the source
-- file is maintained.  Every token has a start and end RealSrcLoc
-- attached to it.
--
-- References:
-- [1] https://www.haskell.org/onlinereport/haskell2010/haskellch2.html
-- [2] http://www.haskell.org/alex/
-- [3] https://gitlab.haskell.org/ghc/ghc/wikis/commentary/compiler/parser
--
-----------------------------------------------------------------------------

--   ToDo / known bugs:
--    - parsing integers is a bit slow
--    - readRational is a bit slow
--
--   Known bugs, that were also in the previous version:
--    - M... should be 3 tokens, not 1.
--    - pragma-end should be only valid in a pragma

--   qualified operator NOTES.
--
--   - If M.(+) is a single lexeme, then..
--     - Probably (+) should be a single lexeme too, for consistency.
--       Otherwise ( + ) would be a prefix operator, but M.( + ) would not be.
--     - But we have to rule out reserved operators, otherwise (..) becomes
--       a different lexeme.
--     - Should we therefore also rule out reserved operators in the qualified
--       form?  This is quite difficult to achieve.  We don't do it for
--       qualified varids.


-- -----------------------------------------------------------------------------
-- Alex "Haskell code fragment top"

{
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiWayIf #-}

{-# OPTIONS_GHC -funbox-strict-fields #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module GHC.Parser.Lexer (
   Token(..), lexer, lexerDbg, pragState, mkPState, mkPStatePure, PState(..),
   P(..), ParseResult(..), mkParserFlags, mkParserFlags', ParserFlags(..),
   appendWarning,
   appendError,
   allocateComments,
   MonadP(..),
   getRealSrcLoc, getPState, withHomeUnitId,
   failMsgP, failLocMsgP, srcParseFail,
   getErrorMessages, getMessages,
   popContext, pushModuleContext, setLastToken, setSrcLoc,
   activeContext, nextIsEOF,
   getLexState, popLexState, pushLexState,
   ExtBits(..),
   xtest,
   lexTokenStream,
   AddAnn(..),mkParensApiAnn,
   addAnnsAt,
   commentToAnnotation
  ) where

import GHC.Prelude

-- base
import Control.Monad
import Data.Bits
import Data.Char
import Data.List
import Data.Maybe
import Data.Word

import GHC.Data.EnumSet as EnumSet

-- ghc-boot
import qualified GHC.LanguageExtensions as LangExt

-- bytestring
import Data.ByteString (ByteString)

-- containers
import Data.Map (Map)
import qualified Data.Map as Map

-- compiler/utils
import GHC.Data.Bag
import GHC.Utils.Outputable
import GHC.Data.StringBuffer
import GHC.Data.FastString
import GHC.Types.Unique.FM
import GHC.Utils.Misc ( readRational, readHexRational )

-- compiler/main
import GHC.Utils.Error
import GHC.Driver.Session as DynFlags

-- compiler/basicTypes
import GHC.Types.SrcLoc
import GHC.Unit
import GHC.Types.Basic ( InlineSpec(..), RuleMatchInfo(..),
                         IntegralLit(..), FractionalLit(..),
                         SourceText(..) )

-- compiler/parser
import GHC.Parser.CharClass

import GHC.Parser.Annotation
}

-- -----------------------------------------------------------------------------
-- Alex "Character set macros"

-- NB: The logic behind these definitions is also reflected in "GHC.Utils.Lexeme"
-- Any changes here should likely be reflected there.
$unispace    = \x05 -- Trick Alex into handling Unicode. See [Unicode in Alex].
$nl          = [\n\r\f]
$whitechar   = [$nl\v\ $unispace]
$white_no_nl = $whitechar # \n -- TODO #8424
$tab         = \t

$ascdigit  = 0-9
$unidigit  = \x03 -- Trick Alex into handling Unicode. See [Unicode in Alex].
$decdigit  = $ascdigit -- for now, should really be $digit (ToDo)
$digit     = [$ascdigit $unidigit]

$special   = [\(\)\,\;\[\]\`\{\}]
$ascsymbol = [\!\#\$\%\&\*\+\.\/\<\=\>\?\@\\\^\|\-\~\:]
$unisymbol = \x04 -- Trick Alex into handling Unicode. See [Unicode in Alex].
$symbol    = [$ascsymbol $unisymbol] # [$special \_\"\']

$unilarge  = \x01 -- Trick Alex into handling Unicode. See [Unicode in Alex].
$asclarge  = [A-Z]
$large     = [$asclarge $unilarge]

$unismall  = \x02 -- Trick Alex into handling Unicode. See [Unicode in Alex].
$ascsmall  = [a-z]
$small     = [$ascsmall $unismall \_]

$unigraphic = \x06 -- Trick Alex into handling Unicode. See [Unicode in Alex].
$graphic   = [$small $large $symbol $digit $special $unigraphic \"\']

$binit     = 0-1
$octit     = 0-7
$hexit     = [$decdigit A-F a-f]

$uniidchar = \x07 -- Trick Alex into handling Unicode. See [Unicode in Alex].
$idchar    = [$small $large $digit $uniidchar \']

$pragmachar = [$small $large $digit]

$docsym    = [\| \^ \* \$]


-- -----------------------------------------------------------------------------
-- Alex "Regular expression macros"

@varid     = $small $idchar*          -- variable identifiers
@conid     = $large $idchar*          -- constructor identifiers

@varsym    = ($symbol # \:) $symbol*  -- variable (operator) symbol
@consym    = \: $symbol*              -- constructor (operator) symbol

-- See Note [Lexing NumericUnderscores extension] and #14473
@numspc       = _*                   -- numeric spacer (#14473)
@decimal      = $decdigit(@numspc $decdigit)*
@binary       = $binit(@numspc $binit)*
@octal        = $octit(@numspc $octit)*
@hexadecimal  = $hexit(@numspc $hexit)*
@exponent     = @numspc [eE] [\-\+]? @decimal
@bin_exponent = @numspc [pP] [\-\+]? @decimal

@qual = (@conid \.)+
@qvarid = @qual @varid
@qconid = @qual @conid
@qvarsym = @qual @varsym
@qconsym = @qual @consym

@floating_point = @numspc @decimal \. @decimal @exponent? | @numspc @decimal @exponent
@hex_floating_point = @numspc @hexadecimal \. @hexadecimal @bin_exponent? | @numspc @hexadecimal @bin_exponent

-- normal signed numerical literals can only be explicitly negative,
-- not explicitly positive (contrast @exponent)
@negative = \-
@signed = @negative ?


-- -----------------------------------------------------------------------------
-- Alex "Identifier"

haskell :-


-- -----------------------------------------------------------------------------
-- Alex "Rules"

-- everywhere: skip whitespace
$white_no_nl+ ;
$tab          { warnTab }

-- Everywhere: deal with nested comments.  We explicitly rule out
-- pragmas, "{-#", so that we don't accidentally treat them as comments.
-- (this can happen even though pragmas will normally take precedence due to
-- longest-match, because pragmas aren't valid in every state, but comments
-- are). We also rule out nested Haddock comments, if the -haddock flag is
-- set.

"{-" / { isNormalComment } { nested_comment lexToken }

-- Single-line comments are a bit tricky.  Haskell 98 says that two or
-- more dashes followed by a symbol should be parsed as a varsym, so we
-- have to exclude those.

-- Since Haddock comments aren't valid in every state, we need to rule them
-- out here.

-- The following two rules match comments that begin with two dashes, but
-- continue with a different character. The rules test that this character
-- is not a symbol (in which case we'd have a varsym), and that it's not a
-- space followed by a Haddock comment symbol (docsym) (in which case we'd
-- have a Haddock comment). The rules then munch the rest of the line.

"-- " ~$docsym .* { lineCommentToken }
"--" [^$symbol \ ] .* { lineCommentToken }

-- Next, match Haddock comments if no -haddock flag

"-- " $docsym .* / { alexNotPred (ifExtension HaddockBit) } { lineCommentToken }

-- Now, when we've matched comments that begin with 2 dashes and continue
-- with a different character, we need to match comments that begin with three
-- or more dashes (which clearly can't be Haddock comments). We only need to
-- make sure that the first non-dash character isn't a symbol, and munch the
-- rest of the line.

"---"\-* ~$symbol .* { lineCommentToken }

-- Since the previous rules all match dashes followed by at least one
-- character, we also need to match a whole line filled with just dashes.

"--"\-* / { atEOL } { lineCommentToken }

-- We need this rule since none of the other single line comment rules
-- actually match this case.

"-- " / { atEOL } { lineCommentToken }

-- 'bol' state: beginning of a line.  Slurp up all the whitespace (including
-- blank lines) until we find a non-whitespace character, then do layout
-- processing.
--
-- One slight wibble here: what if the line begins with {-#? In
-- theory, we have to lex the pragma to see if it's one we recognise,
-- and if it is, then we backtrack and do_bol, otherwise we treat it
-- as a nested comment.  We don't bother with this: if the line begins
-- with {-#, then we'll assume it's a pragma we know about and go for do_bol.
<bol> {
  \n                                    ;
  ^\# line                              { begin line_prag1 }
  ^\# / { followedByDigit }             { begin line_prag1 }
  ^\# pragma .* \n                      ; -- GCC 3.3 CPP generated, apparently
  ^\# \! .* \n                          ; -- #!, for scripts
  ()                                    { do_bol }
}

-- after a layout keyword (let, where, do, of), we begin a new layout
-- context if the curly brace is missing.
-- Careful! This stuff is quite delicate.
<layout, layout_do, layout_if> {
  \{ / { notFollowedBy '-' }            { hopefully_open_brace }
        -- we might encounter {-# here, but {- has been handled already
  \n                                    ;
  ^\# (line)?                           { begin line_prag1 }
}

-- after an 'if', a vertical bar starts a layout context for MultiWayIf
<layout_if> {
  \| / { notFollowedBySymbol }          { new_layout_context True dontGenerateSemic ITvbar }
  ()                                    { pop }
}

-- do is treated in a subtly different way, see new_layout_context
<layout>    ()                          { new_layout_context True  generateSemic ITvocurly }
<layout_do> ()                          { new_layout_context False generateSemic ITvocurly }

-- after a new layout context which was found to be to the left of the
-- previous context, we have generated a '{' token, and we now need to
-- generate a matching '}' token.
<layout_left>  ()                       { do_layout_left }

<0,option_prags> \n                     { begin bol }

"{-#" $whitechar* $pragmachar+ / { known_pragma linePrags }
                                { dispatch_pragmas linePrags }

-- single-line line pragmas, of the form
--    # <line> "<file>" <extra-stuff> \n
<line_prag1> {
  @decimal $white_no_nl+ \" [$graphic \ ]* \"  { setLineAndFile line_prag1a }
  ()                                           { failLinePrag1 }
}
<line_prag1a> .*                               { popLinePrag1 }

-- Haskell-style line pragmas, of the form
--    {-# LINE <line> "<file>" #-}
<line_prag2> {
  @decimal $white_no_nl+ \" [$graphic \ ]* \"  { setLineAndFile line_prag2a }
}
<line_prag2a> "#-}"|"-}"                       { pop }
   -- NOTE: accept -} at the end of a LINE pragma, for compatibility
   -- with older versions of GHC which generated these.

-- Haskell-style column pragmas, of the form
--    {-# COLUMN <column> #-}
<column_prag> @decimal $whitechar* "#-}" { setColumn }

<0,option_prags> {
  "{-#" $whitechar* $pragmachar+
        $whitechar+ $pragmachar+ / { known_pragma twoWordPrags }
                                 { dispatch_pragmas twoWordPrags }

  "{-#" $whitechar* $pragmachar+ / { known_pragma oneWordPrags }
                                 { dispatch_pragmas oneWordPrags }

  -- We ignore all these pragmas, but don't generate a warning for them
  "{-#" $whitechar* $pragmachar+ / { known_pragma ignoredPrags }
                                 { dispatch_pragmas ignoredPrags }

  -- ToDo: should only be valid inside a pragma:
  "#-}"                          { endPrag }
}

<option_prags> {
  "{-#"  $whitechar* $pragmachar+ / { known_pragma fileHeaderPrags }
                                   { dispatch_pragmas fileHeaderPrags }
}

<0> {
  -- In the "0" mode we ignore these pragmas
  "{-#"  $whitechar* $pragmachar+ / { known_pragma fileHeaderPrags }
                     { nested_comment lexToken }
}

<0,option_prags> {
  "{-#"  { warnThen Opt_WarnUnrecognisedPragmas (text "Unrecognised pragma")
                    (nested_comment lexToken) }
}

-- '0' state: ordinary lexemes

-- Haddock comments

<0,option_prags> {
  "-- " $docsym      / { ifExtension HaddockBit } { multiline_doc_comment }
  "{-" \ ? $docsym   / { ifExtension HaddockBit } { nested_doc_comment }
}

-- "special" symbols

<0> {

  -- Don't check ThQuotesBit here as the renamer can produce a better
  -- error message than the lexer (see the thQuotesEnabled check in rnBracket).
  "[|"  { token (ITopenExpQuote NoE NormalSyntax) }
  "[||" { token (ITopenTExpQuote NoE) }
  "|]"  { token (ITcloseQuote NormalSyntax) }
  "||]" { token ITcloseTExpQuote }

  -- Check ThQuotesBit here as to not steal syntax.
  "[e|"       / { ifExtension ThQuotesBit } { token (ITopenExpQuote HasE NormalSyntax) }
  "[e||"      / { ifExtension ThQuotesBit } { token (ITopenTExpQuote HasE) }
  "[p|"       / { ifExtension ThQuotesBit } { token ITopenPatQuote }
  "[d|"       / { ifExtension ThQuotesBit } { layout_token ITopenDecQuote }
  "[t|"       / { ifExtension ThQuotesBit } { token ITopenTypQuote }

  "[" @varid "|"  / { ifExtension QqBit }   { lex_quasiquote_tok }

  -- qualified quasi-quote (#5555)
  "[" @qvarid "|"  / { ifExtension QqBit }  { lex_qquasiquote_tok }

  $unigraphic -- ⟦
    / { ifCurrentChar '⟦' `alexAndPred`
        ifExtension UnicodeSyntaxBit `alexAndPred`
        ifExtension ThQuotesBit }
    { token (ITopenExpQuote NoE UnicodeSyntax) }
  $unigraphic -- ⟧
    / { ifCurrentChar '⟧' `alexAndPred`
        ifExtension UnicodeSyntaxBit `alexAndPred`
        ifExtension ThQuotesBit }
    { token (ITcloseQuote UnicodeSyntax) }
}

<0> {
  "(|"
    / { ifExtension ArrowsBit `alexAndPred`
        notFollowedBySymbol }
    { special (IToparenbar NormalSyntax) }
  "|)"
    / { ifExtension ArrowsBit }
    { special (ITcparenbar NormalSyntax) }

  $unigraphic -- ⦇
    / { ifCurrentChar '⦇' `alexAndPred`
        ifExtension UnicodeSyntaxBit `alexAndPred`
        ifExtension ArrowsBit }
    { special (IToparenbar UnicodeSyntax) }
  $unigraphic -- ⦈
    / { ifCurrentChar '⦈' `alexAndPred`
        ifExtension UnicodeSyntaxBit `alexAndPred`
        ifExtension ArrowsBit }
    { special (ITcparenbar UnicodeSyntax) }
}

<0> {
  \? @varid / { ifExtension IpBit } { skip_one_varid ITdupipvarid }
}

<0> {
  "#" @varid / { ifExtension OverloadedLabelsBit } { skip_one_varid ITlabelvarid }
}

<0> {
  "(#" / { ifExtension UnboxedTuplesBit `alexOrPred`
           ifExtension UnboxedSumsBit }
         { token IToubxparen }
  "#)" / { ifExtension UnboxedTuplesBit `alexOrPred`
           ifExtension UnboxedSumsBit }
         { token ITcubxparen }
}

<0,option_prags> {
  \(                                    { special IToparen }
  \)                                    { special ITcparen }
  \[                                    { special ITobrack }
  \]                                    { special ITcbrack }
  \,                                    { special ITcomma }
  \;                                    { special ITsemi }
  \`                                    { special ITbackquote }

  \{                                    { open_brace }
  \}                                    { close_brace }
}

<0,option_prags> {
  @qvarid                       { idtoken qvarid }
  @qconid                       { idtoken qconid }
  @varid                        { varid }
  @conid                        { idtoken conid }
}

<0> {
  @qvarid "#"+      / { ifExtension MagicHashBit } { idtoken qvarid }
  @qconid "#"+      / { ifExtension MagicHashBit } { idtoken qconid }
  @varid "#"+       / { ifExtension MagicHashBit } { varid }
  @conid "#"+       / { ifExtension MagicHashBit } { idtoken conid }
}

-- Operators classified into prefix, suffix, tight infix, and loose infix.
-- See Note [Whitespace-sensitive operator parsing]
<0> {
  @varsym / { precededByClosingToken `alexAndPred` followedByOpeningToken } { varsym_tight_infix }
  @varsym / { followedByOpeningToken }  { varsym_prefix }
  @varsym / { precededByClosingToken }  { varsym_suffix }
  @varsym                               { varsym_loose_infix }
}

-- ToDo: - move `var` and (sym) into lexical syntax?
--       - remove backquote from $special?
<0> {
  @qvarsym                                         { idtoken qvarsym }
  @qconsym                                         { idtoken qconsym }
  @consym                                          { consym }
}

-- For the normal boxed literals we need to be careful
-- when trying to be close to Haskell98

-- Note [Lexing NumericUnderscores extension] (#14473)
--
-- NumericUnderscores extension allows underscores in numeric literals.
-- Multiple underscores are represented with @numspc macro.
-- To be simpler, we have only the definitions with underscores.
-- And then we have a separate function (tok_integral and tok_frac)
-- that validates the literals.
-- If extensions are not enabled, check that there are no underscores.
--
<0> {
  -- Normal integral literals (:: Num a => a, from Integer)
  @decimal                                                                   { tok_num positive 0 0 decimal }
  0[bB] @numspc @binary                / { ifExtension BinaryLiteralsBit }   { tok_num positive 2 2 binary }
  0[oO] @numspc @octal                                                       { tok_num positive 2 2 octal }
  0[xX] @numspc @hexadecimal                                                 { tok_num positive 2 2 hexadecimal }
  @negative @decimal                   / { ifExtension NegativeLiteralsBit } { tok_num negative 1 1 decimal }
  @negative 0[bB] @numspc @binary      / { ifExtension NegativeLiteralsBit `alexAndPred`
                                           ifExtension BinaryLiteralsBit }   { tok_num negative 3 3 binary }
  @negative 0[oO] @numspc @octal       / { ifExtension NegativeLiteralsBit } { tok_num negative 3 3 octal }
  @negative 0[xX] @numspc @hexadecimal / { ifExtension NegativeLiteralsBit } { tok_num negative 3 3 hexadecimal }

  -- Normal rational literals (:: Fractional a => a, from Rational)
  @floating_point                                                            { tok_frac 0 tok_float }
  @negative @floating_point            / { ifExtension NegativeLiteralsBit } { tok_frac 0 tok_float }
  0[xX] @numspc @hex_floating_point    / { ifExtension HexFloatLiteralsBit } { tok_frac 0 tok_hex_float }
  @negative 0[xX] @numspc @hex_floating_point
                                       / { ifExtension HexFloatLiteralsBit `alexAndPred`
                                           ifExtension NegativeLiteralsBit } { tok_frac 0 tok_hex_float }
}

<0> {
  -- Unboxed ints (:: Int#) and words (:: Word#)
  -- It's simpler (and faster?) to give separate cases to the negatives,
  -- especially considering octal/hexadecimal prefixes.
  @decimal                          \# / { ifExtension MagicHashBit }        { tok_primint positive 0 1 decimal }
  0[bB] @numspc @binary             \# / { ifExtension MagicHashBit `alexAndPred`
                                           ifExtension BinaryLiteralsBit }   { tok_primint positive 2 3 binary }
  0[oO] @numspc @octal              \# / { ifExtension MagicHashBit }        { tok_primint positive 2 3 octal }
  0[xX] @numspc @hexadecimal        \# / { ifExtension MagicHashBit }        { tok_primint positive 2 3 hexadecimal }
  @negative @decimal                \# / { ifExtension MagicHashBit }        { tok_primint negative 1 2 decimal }
  @negative 0[bB] @numspc @binary   \# / { ifExtension MagicHashBit `alexAndPred`
                                           ifExtension BinaryLiteralsBit }   { tok_primint negative 3 4 binary }
  @negative 0[oO] @numspc @octal    \# / { ifExtension MagicHashBit }        { tok_primint negative 3 4 octal }
  @negative 0[xX] @numspc @hexadecimal \#
                                       / { ifExtension MagicHashBit }        { tok_primint negative 3 4 hexadecimal }

  @decimal                       \# \# / { ifExtension MagicHashBit }        { tok_primword 0 2 decimal }
  0[bB] @numspc @binary          \# \# / { ifExtension MagicHashBit `alexAndPred`
                                           ifExtension BinaryLiteralsBit }   { tok_primword 2 4 binary }
  0[oO] @numspc @octal           \# \# / { ifExtension MagicHashBit }        { tok_primword 2 4 octal }
  0[xX] @numspc @hexadecimal     \# \# / { ifExtension MagicHashBit }        { tok_primword 2 4 hexadecimal }

  -- Unboxed floats and doubles (:: Float#, :: Double#)
  -- prim_{float,double} work with signed literals
  @signed @floating_point           \# / { ifExtension MagicHashBit }        { tok_frac 1 tok_primfloat }
  @signed @floating_point        \# \# / { ifExtension MagicHashBit }        { tok_frac 2 tok_primdouble }
}

-- Strings and chars are lexed by hand-written code.  The reason is
-- that even if we recognise the string or char here in the regex
-- lexer, we would still have to parse the string afterward in order
-- to convert it to a String.
<0> {
  \'                            { lex_char_tok }
  \"                            { lex_string_tok }
}

-- Note [Whitespace-sensitive operator parsing]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- In accord with GHC Proposal #229 https://github.com/ghc-proposals/ghc-proposals/blob/master/proposals/0229-whitespace-bang-patterns.rst
-- we classify operator occurrences into four categories:
--
--     a ! b   -- a loose infix occurrence
--     a!b     -- a tight infix occurrence
--     a !b    -- a prefix occurrence
--     a! b    -- a suffix occurrence
--
-- The rules are a bit more elaborate than simply checking for whitespace, in
-- order to accommodate the following use cases:
--
--     f (!a) = ...    -- prefix occurrence
--     g (a !)         -- loose infix occurrence
--     g (! a)         -- loose infix occurrence
--
-- The precise rules are as follows:
--
--  * Identifiers, literals, and opening brackets (, (#, (|, [, [|, [||, [p|,
--    [e|, [t|, {, ⟦, ⦇, are considered "opening tokens". The function
--    followedByOpeningToken tests whether the next token is an opening token.
--
--  * Identifiers, literals, and closing brackets ), #), |), ], |], }, ⟧, ⦈,
--    are considered "closing tokens". The function precededByClosingToken tests
--    whether the previous token is a closing token.
--
--  * Whitespace, comments, separators, and other tokens, are considered
--    neither opening nor closing.
--
--  * Any unqualified operator occurrence is classified as prefix, suffix, or
--    tight/loose infix, based on preceding and following tokens:
--
--       precededByClosingToken | followedByOpeningToken | Occurrence
--      ------------------------+------------------------+------------
--       False                  | True                   | prefix
--       True                   | False                  | suffix
--       True                   | True                   | tight infix
--       False                  | False                  | loose infix
--      ------------------------+------------------------+------------
--
-- A loose infix occurrence is always considered an operator. Other types of
-- occurrences may be assigned a special per-operator meaning override:
--
--   Operator |  Occurrence   | Token returned
--  ----------+---------------+------------------------------------------
--    !       |  prefix       | ITbang
--            |               |   strictness annotation or bang pattern,
--            |               |   e.g.  f !x = rhs, data T = MkT !a
--            |  not prefix   | ITvarsym "!"
--            |               |   ordinary operator or type operator,
--            |               |   e.g.  xs ! 3, (! x), Int ! Bool
--  ----------+---------------+------------------------------------------
--    ~       |  prefix       | ITtilde
--            |               |   laziness annotation or lazy pattern,
--            |               |   e.g.  f ~x = rhs, data T = MkT ~a
--            |  not prefix   | ITvarsym "~"
--            |               |   ordinary operator or type operator,
--            |               |   e.g.  xs ~ 3, (~ x), Int ~ Bool
--  ----------+---------------+------------------------------------------
--    $  $$   |  prefix       | ITdollar, ITdollardollar
--            |               |   untyped or typed Template Haskell splice,
--            |               |   e.g.  $(f x), $$(f x), $$"str"
--            |  not prefix   | ITvarsym "$", ITvarsym "$$"
--            |               |   ordinary operator or type operator,
--            |               |   e.g.  f $ g x, a $$ b
--  ----------+---------------+------------------------------------------
--    @       |  prefix       | ITtypeApp
--            |               |   type application, e.g.  fmap @Maybe
--            |  tight infix  | ITat
--            |               |   as-pattern, e.g.  f p@(a,b) = rhs
--            |  suffix       | parse error
--            |               |   e.g. f p@ x = rhs
--            |  loose infix  | ITvarsym "@"
--            |               |   ordinary operator or type operator,
--            |               |   e.g.  f @ g, (f @)
--  ----------+---------------+------------------------------------------
--
-- Also, some of these overrides are guarded behind language extensions.
-- According to the specification, we must determine the occurrence based on
-- surrounding *tokens* (see the proposal for the exact rules). However, in
-- the implementation we cheat a little and do the classification based on
-- characters, for reasons of both simplicity and efficiency (see
-- 'followedByOpeningToken' and 'precededByClosingToken')
--
-- When an operator is subject to a meaning override, it is mapped to special
-- token: ITbang, ITtilde, ITat, ITdollar, ITdollardollar. Otherwise, it is
-- returned as ITvarsym.
--
-- For example, this is how we process the (!):
--
--    precededByClosingToken | followedByOpeningToken | Token
--   ------------------------+------------------------+-------------
--    False                  | True                   | ITbang
--    True                   | False                  | ITvarsym "!"
--    True                   | True                   | ITvarsym "!"
--    False                  | False                  | ITvarsym "!"
--   ------------------------+------------------------+-------------
--
-- And this is how we process the (@):
--
--    precededByClosingToken | followedByOpeningToken | Token
--   ------------------------+------------------------+-------------
--    False                  | True                   | ITtypeApp
--    True                   | False                  | parse error
--    True                   | True                   | ITat
--    False                  | False                  | ITvarsym "@"
--   ------------------------+------------------------+-------------

-- -----------------------------------------------------------------------------
-- Alex "Haskell code fragment bottom"

{

-- -----------------------------------------------------------------------------
-- The token type

data Token
  = ITas                        -- Haskell keywords
  | ITcase
  | ITclass
  | ITdata
  | ITdefault
  | ITderiving
  | ITdo
  | ITelse
  | IThiding
  | ITforeign
  | ITif
  | ITimport
  | ITin
  | ITinfix
  | ITinfixl
  | ITinfixr
  | ITinstance
  | ITlet
  | ITmodule
  | ITnewtype
  | ITof
  | ITqualified
  | ITthen
  | ITtype
  | ITwhere

  | ITforall            IsUnicodeSyntax -- GHC extension keywords
  | ITexport
  | ITlabel
  | ITdynamic
  | ITsafe
  | ITinterruptible
  | ITunsafe
  | ITstdcallconv
  | ITccallconv
  | ITcapiconv
  | ITprimcallconv
  | ITjavascriptcallconv
  | ITmdo
  | ITfamily
  | ITrole
  | ITgroup
  | ITby
  | ITusing
  | ITpattern
  | ITstatic
  | ITstock
  | ITanyclass
  | ITvia

  -- Backpack tokens
  | ITunit
  | ITsignature
  | ITdependency
  | ITrequires

  -- Pragmas, see  note [Pragma source text] in "GHC.Types.Basic"
  | ITinline_prag       SourceText InlineSpec RuleMatchInfo
  | ITspec_prag         SourceText                -- SPECIALISE
  | ITspec_inline_prag  SourceText Bool    -- SPECIALISE INLINE (or NOINLINE)
  | ITsource_prag       SourceText
  | ITrules_prag        SourceText
  | ITwarning_prag      SourceText
  | ITdeprecated_prag   SourceText
  | ITline_prag         SourceText  -- not usually produced, see 'UsePosPragsBit'
  | ITcolumn_prag       SourceText  -- not usually produced, see 'UsePosPragsBit'
  | ITscc_prag          SourceText
  | ITgenerated_prag    SourceText
  | ITcore_prag         SourceText         -- hdaume: core annotations
  | ITunpack_prag       SourceText
  | ITnounpack_prag     SourceText
  | ITann_prag          SourceText
  | ITcomplete_prag     SourceText
  | ITclose_prag
  | IToptions_prag String
  | ITinclude_prag String
  | ITlanguage_prag
  | ITminimal_prag      SourceText
  | IToverlappable_prag SourceText  -- instance overlap mode
  | IToverlapping_prag  SourceText  -- instance overlap mode
  | IToverlaps_prag     SourceText  -- instance overlap mode
  | ITincoherent_prag   SourceText  -- instance overlap mode
  | ITctype             SourceText
  | ITcomment_line_prag         -- See Note [Nested comment line pragmas]

  | ITdotdot                    -- reserved symbols
  | ITcolon
  | ITdcolon            IsUnicodeSyntax
  | ITequal
  | ITlam
  | ITlcase
  | ITvbar
  | ITlarrow            IsUnicodeSyntax
  | ITrarrow            IsUnicodeSyntax
  | ITlolly             IsUnicodeSyntax
  | ITdarrow            IsUnicodeSyntax
  | ITminus       -- See Note [Minus tokens]
  | ITprefixminus -- See Note [Minus tokens]
  | ITbang     -- Prefix (!) only, e.g. f !x = rhs
  | ITtilde    -- Prefix (~) only, e.g. f ~x = rhs
  | ITat       -- Tight infix (@) only, e.g. f x@pat = rhs
  | ITtypeApp  -- Prefix (@) only, e.g. f @t
  | ITstar              IsUnicodeSyntax
  | ITdot

  | ITbiglam                    -- GHC-extension symbols

  | ITocurly                    -- special symbols
  | ITccurly
  | ITvocurly
  | ITvccurly
  | ITobrack
  | ITopabrack                  -- [:, for parallel arrays with -XParallelArrays
  | ITcpabrack                  -- :], for parallel arrays with -XParallelArrays
  | ITcbrack
  | IToparen
  | ITcparen
  | IToubxparen
  | ITcubxparen
  | ITsemi
  | ITcomma
  | ITunderscore
  | ITbackquote
  | ITsimpleQuote               --  '

  | ITvarid   FastString        -- identifiers
  | ITconid   FastString
  | ITvarsym  FastString
  | ITconsym  FastString
  | ITqvarid  (FastString,FastString)
  | ITqconid  (FastString,FastString)
  | ITqvarsym (FastString,FastString)
  | ITqconsym (FastString,FastString)

  | ITdupipvarid   FastString   -- GHC extension: implicit param: ?x
  | ITlabelvarid   FastString   -- Overloaded label: #x

  | ITchar     SourceText Char       -- Note [Literal source text] in "GHC.Types.Basic"
  | ITstring   SourceText FastString -- Note [Literal source text] in "GHC.Types.Basic"
  | ITinteger  IntegralLit           -- Note [Literal source text] in "GHC.Types.Basic"
  | ITrational FractionalLit

  | ITprimchar   SourceText Char     -- Note [Literal source text] in "GHC.Types.Basic"
  | ITprimstring SourceText ByteString -- Note [Literal source text] in "GHC.Types.Basic"
  | ITprimint    SourceText Integer  -- Note [Literal source text] in "GHC.Types.Basic"
  | ITprimword   SourceText Integer  -- Note [Literal source text] in "GHC.Types.Basic"
  | ITprimfloat  FractionalLit
  | ITprimdouble FractionalLit

  -- Template Haskell extension tokens
  | ITopenExpQuote HasE IsUnicodeSyntax --  [| or [e|
  | ITopenPatQuote                      --  [p|
  | ITopenDecQuote                      --  [d|
  | ITopenTypQuote                      --  [t|
  | ITcloseQuote IsUnicodeSyntax        --  |]
  | ITopenTExpQuote HasE                --  [|| or [e||
  | ITcloseTExpQuote                    --  ||]
  | ITdollar                            --  prefix $
  | ITdollardollar                      --  prefix $$
  | ITtyQuote                           --  ''
  | ITquasiQuote (FastString,FastString,PsSpan)
    -- ITquasiQuote(quoter, quote, loc)
    -- represents a quasi-quote of the form
    -- [quoter| quote |]
  | ITqQuasiQuote (FastString,FastString,FastString,PsSpan)
    -- ITqQuasiQuote(Qual, quoter, quote, loc)
    -- represents a qualified quasi-quote of the form
    -- [Qual.quoter| quote |]

  -- Arrow notation extension
  | ITproc
  | ITrec
  | IToparenbar  IsUnicodeSyntax -- ^ @(|@
  | ITcparenbar  IsUnicodeSyntax -- ^ @|)@
  | ITlarrowtail IsUnicodeSyntax -- ^ @-<@
  | ITrarrowtail IsUnicodeSyntax -- ^ @>-@
  | ITLarrowtail IsUnicodeSyntax -- ^ @-<<@
  | ITRarrowtail IsUnicodeSyntax -- ^ @>>-@

  | ITunknown String             -- ^ Used when the lexer can't make sense of it
  | ITeof                        -- ^ end of file token

  -- Documentation annotations
  | ITdocCommentNext  String     -- ^ something beginning @-- |@
  | ITdocCommentPrev  String     -- ^ something beginning @-- ^@
  | ITdocCommentNamed String     -- ^ something beginning @-- $@
  | ITdocSection      Int String -- ^ a section heading
  | ITdocOptions      String     -- ^ doc options (prune, ignore-exports, etc)
  | ITlineComment     String     -- ^ comment starting by "--"
  | ITblockComment    String     -- ^ comment in {- -}

  deriving Show

instance Outputable Token where
  ppr x = text (show x)


{- Note [Minus tokens]
~~~~~~~~~~~~~~~~~~~~~~
A minus sign can be used in prefix form (-x) and infix form (a - b).

When LexicalNegation is on:
  * ITprefixminus  represents the prefix form
  * ITvarsym "-"   represents the infix form
  * ITminus        is not used

When LexicalNegation is off:
  * ITminus        represents all forms
  * ITprefixminus  is not used
  * ITvarsym "-"   is not used
-}

{- Note [Why not LexicalNegationBit]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One might wonder why we define NoLexicalNegationBit instead of
LexicalNegationBit. The problem lies in the following line in reservedSymsFM:

    ,("-", ITminus, NormalSyntax, xbit NoLexicalNegationBit)

We want to generate ITminus only when LexicalNegation is off. How would one
do it if we had LexicalNegationBit? I (int-index) tried to use bitwise
complement:

    ,("-", ITminus, NormalSyntax, complement (xbit LexicalNegationBit))

This did not work, so I opted for NoLexicalNegationBit instead.
-}


-- the bitmap provided as the third component indicates whether the
-- corresponding extension keyword is valid under the extension options
-- provided to the compiler; if the extension corresponding to *any* of the
-- bits set in the bitmap is enabled, the keyword is valid (this setup
-- facilitates using a keyword in two different extensions that can be
-- activated independently)
--
reservedWordsFM :: UniqFM (Token, ExtsBitmap)
reservedWordsFM = listToUFM $
    map (\(x, y, z) -> (mkFastString x, (y, z)))
        [( "_",              ITunderscore,    0 ),
         ( "as",             ITas,            0 ),
         ( "case",           ITcase,          0 ),
         ( "class",          ITclass,         0 ),
         ( "data",           ITdata,          0 ),
         ( "default",        ITdefault,       0 ),
         ( "deriving",       ITderiving,      0 ),
         ( "do",             ITdo,            0 ),
         ( "else",           ITelse,          0 ),
         ( "hiding",         IThiding,        0 ),
         ( "if",             ITif,            0 ),
         ( "import",         ITimport,        0 ),
         ( "in",             ITin,            0 ),
         ( "infix",          ITinfix,         0 ),
         ( "infixl",         ITinfixl,        0 ),
         ( "infixr",         ITinfixr,        0 ),
         ( "instance",       ITinstance,      0 ),
         ( "let",            ITlet,           0 ),
         ( "module",         ITmodule,        0 ),
         ( "newtype",        ITnewtype,       0 ),
         ( "of",             ITof,            0 ),
         ( "qualified",      ITqualified,     0 ),
         ( "then",           ITthen,          0 ),
         ( "type",           ITtype,          0 ),
         ( "where",          ITwhere,         0 ),

         ( "forall",         ITforall NormalSyntax, 0),
         ( "mdo",            ITmdo,           xbit RecursiveDoBit),
             -- See Note [Lexing type pseudo-keywords]
         ( "family",         ITfamily,        0 ),
         ( "role",           ITrole,          0 ),
         ( "pattern",        ITpattern,       xbit PatternSynonymsBit),
         ( "static",         ITstatic,        xbit StaticPointersBit ),
         ( "stock",          ITstock,         0 ),
         ( "anyclass",       ITanyclass,      0 ),
         ( "via",            ITvia,           0 ),
         ( "group",          ITgroup,         xbit TransformComprehensionsBit),
         ( "by",             ITby,            xbit TransformComprehensionsBit),
         ( "using",          ITusing,         xbit TransformComprehensionsBit),

         ( "foreign",        ITforeign,       xbit FfiBit),
         ( "export",         ITexport,        xbit FfiBit),
         ( "label",          ITlabel,         xbit FfiBit),
         ( "dynamic",        ITdynamic,       xbit FfiBit),
         ( "safe",           ITsafe,          xbit FfiBit .|.
                                              xbit SafeHaskellBit),
         ( "interruptible",  ITinterruptible, xbit InterruptibleFfiBit),
         ( "unsafe",         ITunsafe,        xbit FfiBit),
         ( "stdcall",        ITstdcallconv,   xbit FfiBit),
         ( "ccall",          ITccallconv,     xbit FfiBit),
         ( "capi",           ITcapiconv,      xbit CApiFfiBit),
         ( "prim",           ITprimcallconv,  xbit FfiBit),
         ( "javascript",     ITjavascriptcallconv, xbit FfiBit),

         ( "unit",           ITunit,          0 ),
         ( "dependency",     ITdependency,       0 ),
         ( "signature",      ITsignature,     0 ),

         ( "rec",            ITrec,           xbit ArrowsBit .|.
                                              xbit RecursiveDoBit),
         ( "proc",           ITproc,          xbit ArrowsBit)
     ]

{-----------------------------------
Note [Lexing type pseudo-keywords]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One might think that we wish to treat 'family' and 'role' as regular old
varids whenever -XTypeFamilies and -XRoleAnnotations are off, respectively.
But, there is no need to do so. These pseudo-keywords are not stolen syntax:
they are only used after the keyword 'type' at the top-level, where varids are
not allowed. Furthermore, checks further downstream (GHC.Tc.TyCl) ensure that
type families and role annotations are never declared without their extensions
on. In fact, by unconditionally lexing these pseudo-keywords as special, we
can get better error messages.

Also, note that these are included in the `varid` production in the parser --
a key detail to make all this work.
-------------------------------------}

reservedSymsFM :: UniqFM (Token, IsUnicodeSyntax, ExtsBitmap)
reservedSymsFM = listToUFM $
    map (\ (x,w,y,z) -> (mkFastString x,(w,y,z)))
      [ ("..",  ITdotdot,                   NormalSyntax,  0 )
        -- (:) is a reserved op, meaning only list cons
       ,(":",   ITcolon,                    NormalSyntax,  0 )
       ,("::",  ITdcolon NormalSyntax,      NormalSyntax,  0 )
       ,("=",   ITequal,                    NormalSyntax,  0 )
       ,("\\",  ITlam,                      NormalSyntax,  0 )
       ,("|",   ITvbar,                     NormalSyntax,  0 )
       ,("<-",  ITlarrow NormalSyntax,      NormalSyntax,  0 )
       ,("->",  ITrarrow NormalSyntax,      NormalSyntax,  0 )
       ,("=>",  ITdarrow NormalSyntax,      NormalSyntax,  0 )
       ,("-",   ITminus,                    NormalSyntax,  xbit NoLexicalNegationBit)

       ,("*",   ITstar NormalSyntax,        NormalSyntax,  xbit StarIsTypeBit)

        -- For 'forall a . t'
       ,(".",   ITdot,                      NormalSyntax,  0 )

       ,("-<",  ITlarrowtail NormalSyntax,  NormalSyntax,  xbit ArrowsBit)
       ,(">-",  ITrarrowtail NormalSyntax,  NormalSyntax,  xbit ArrowsBit)
       ,("-<<", ITLarrowtail NormalSyntax,  NormalSyntax,  xbit ArrowsBit)
       ,(">>-", ITRarrowtail NormalSyntax,  NormalSyntax,  xbit ArrowsBit)

       ,("∷",   ITdcolon UnicodeSyntax,     UnicodeSyntax, 0 )
       ,("⇒",   ITdarrow UnicodeSyntax,     UnicodeSyntax, 0 )
       ,("∀",   ITforall UnicodeSyntax,     UnicodeSyntax, 0 )
       ,("→",   ITrarrow UnicodeSyntax,     UnicodeSyntax, 0 )
       ,("←",   ITlarrow UnicodeSyntax,     UnicodeSyntax, 0 )

       ,("#->", ITlolly NormalSyntax, NormalSyntax, 0)
       ,("⊸",   ITlolly UnicodeSyntax, UnicodeSyntax, 0)

       ,("⤙",   ITlarrowtail UnicodeSyntax, UnicodeSyntax, xbit ArrowsBit)
       ,("⤚",   ITrarrowtail UnicodeSyntax, UnicodeSyntax, xbit ArrowsBit)
       ,("⤛",   ITLarrowtail UnicodeSyntax, UnicodeSyntax, xbit ArrowsBit)
       ,("⤜",   ITRarrowtail UnicodeSyntax, UnicodeSyntax, xbit ArrowsBit)

       ,("★",   ITstar UnicodeSyntax,       UnicodeSyntax, xbit StarIsTypeBit)

        -- ToDo: ideally, → and ∷ should be "specials", so that they cannot
        -- form part of a large operator.  This would let us have a better
        -- syntax for kinds: ɑ∷*→* would be a legal kind signature. (maybe).
       ]

-- -----------------------------------------------------------------------------
-- Lexer actions

type Action = PsSpan -> StringBuffer -> Int -> P (PsLocated Token)

special :: Token -> Action
special tok span _buf _len = return (L span tok)

token, layout_token :: Token -> Action
token t span _buf _len = return (L span t)
layout_token t span _buf _len = pushLexState layout >> return (L span t)

idtoken :: (StringBuffer -> Int -> Token) -> Action
idtoken f span buf len = return (L span $! (f buf len))

skip_one_varid :: (FastString -> Token) -> Action
skip_one_varid f span buf len
  = return (L span $! f (lexemeToFastString (stepOn buf) (len-1)))

skip_two_varid :: (FastString -> Token) -> Action
skip_two_varid f span buf len
  = return (L span $! f (lexemeToFastString (stepOn (stepOn buf)) (len-2)))

strtoken :: (String -> Token) -> Action
strtoken f span buf len =
  return (L span $! (f $! lexemeToString buf len))

begin :: Int -> Action
begin code _span _str _len = do pushLexState code; lexToken

pop :: Action
pop _span _buf _len = do _ <- popLexState
                         lexToken
-- See Note [Nested comment line pragmas]
failLinePrag1 :: Action
failLinePrag1 span _buf _len = do
  b <- getBit InNestedCommentBit
  if b then return (L span ITcomment_line_prag)
       else lexError "lexical error in pragma"

-- See Note [Nested comment line pragmas]
popLinePrag1 :: Action
popLinePrag1 span _buf _len = do
  b <- getBit InNestedCommentBit
  if b then return (L span ITcomment_line_prag) else do
    _ <- popLexState
    lexToken

hopefully_open_brace :: Action
hopefully_open_brace span buf len
 = do relaxed <- getBit RelaxedLayoutBit
      ctx <- getContext
      (AI l _) <- getInput
      let offset = srcLocCol (psRealLoc l)
          isOK = relaxed ||
                 case ctx of
                 Layout prev_off _ : _ -> prev_off < offset
                 _                     -> True
      if isOK then pop_and open_brace span buf len
              else addFatalError (mkSrcSpanPs span) (text "Missing block")

pop_and :: Action -> Action
pop_and act span buf len = do _ <- popLexState
                              act span buf len

-- See Note [Whitespace-sensitive operator parsing]
followedByOpeningToken :: AlexAccPred ExtsBitmap
followedByOpeningToken _ _ _ (AI _ buf)
  | atEnd buf = False
  | otherwise =
      case nextChar buf of
        ('{', buf') -> nextCharIsNot buf' (== '-')
        ('(', _) -> True
        ('[', _) -> True
        ('\"', _) -> True
        ('\'', _) -> True
        ('_', _) -> True
        ('⟦', _) -> True
        ('⦇', _) -> True
        (c, _) -> isAlphaNum c

-- See Note [Whitespace-sensitive operator parsing]
precededByClosingToken :: AlexAccPred ExtsBitmap
precededByClosingToken _ (AI _ buf) _ _ =
  case prevChar buf '\n' of
    '}' -> decodePrevNChars 1 buf /= "-"
    ')' -> True
    ']' -> True
    '\"' -> True
    '\'' -> True
    '_' -> True
    '⟧' -> True
    '⦈' -> True
    c -> isAlphaNum c

{-# INLINE nextCharIs #-}
nextCharIs :: StringBuffer -> (Char -> Bool) -> Bool
nextCharIs buf p = not (atEnd buf) && p (currentChar buf)

{-# INLINE nextCharIsNot #-}
nextCharIsNot :: StringBuffer -> (Char -> Bool) -> Bool
nextCharIsNot buf p = not (nextCharIs buf p)

notFollowedBy :: Char -> AlexAccPred ExtsBitmap
notFollowedBy char _ _ _ (AI _ buf)
  = nextCharIsNot buf (== char)

notFollowedBySymbol :: AlexAccPred ExtsBitmap
notFollowedBySymbol _ _ _ (AI _ buf)
  = nextCharIsNot buf (`elem` "!#$%&*+./<=>?@\\^|-~")

followedByDigit :: AlexAccPred ExtsBitmap
followedByDigit _ _ _ (AI _ buf)
  = afterOptionalSpace buf (\b -> nextCharIs b (`elem` ['0'..'9']))

ifCurrentChar :: Char -> AlexAccPred ExtsBitmap
ifCurrentChar char _ (AI _ buf) _ _
  = nextCharIs buf (== char)

-- We must reject doc comments as being ordinary comments everywhere.
-- In some cases the doc comment will be selected as the lexeme due to
-- maximal munch, but not always, because the nested comment rule is
-- valid in all states, but the doc-comment rules are only valid in
-- the non-layout states.
isNormalComment :: AlexAccPred ExtsBitmap
isNormalComment bits _ _ (AI _ buf)
  | HaddockBit `xtest` bits = notFollowedByDocOrPragma
  | otherwise               = nextCharIsNot buf (== '#')
  where
    notFollowedByDocOrPragma
       = afterOptionalSpace buf (\b -> nextCharIsNot b (`elem` "|^*$#"))

afterOptionalSpace :: StringBuffer -> (StringBuffer -> Bool) -> Bool
afterOptionalSpace buf p
    = if nextCharIs buf (== ' ')
      then p (snd (nextChar buf))
      else p buf

atEOL :: AlexAccPred ExtsBitmap
atEOL _ _ _ (AI _ buf) = atEnd buf || currentChar buf == '\n'

ifExtension :: ExtBits -> AlexAccPred ExtsBitmap
ifExtension extBits bits _ _ _ = extBits `xtest` bits

alexNotPred p userState in1 len in2
  = not (p userState in1 len in2)

alexOrPred p1 p2 userState in1 len in2
  = p1 userState in1 len in2 || p2 userState in1 len in2

multiline_doc_comment :: Action
multiline_doc_comment span buf _len = withLexedDocType (worker "")
  where
    worker commentAcc input docType checkNextLine = case alexGetChar' input of
      Just ('\n', input')
        | checkNextLine -> case checkIfCommentLine input' of
          Just input -> worker ('\n':commentAcc) input docType checkNextLine
          Nothing -> docCommentEnd input commentAcc docType buf span
        | otherwise -> docCommentEnd input commentAcc docType buf span
      Just (c, input) -> worker (c:commentAcc) input docType checkNextLine
      Nothing -> docCommentEnd input commentAcc docType buf span

    -- Check if the next line of input belongs to this doc comment as well.
    -- A doc comment continues onto the next line when the following
    -- conditions are met:
    --   * The line starts with "--"
    --   * The line doesn't start with "---".
    --   * The line doesn't start with "-- $", because that would be the
    --     start of a /new/ named haddock chunk (#10398).
    checkIfCommentLine :: AlexInput -> Maybe AlexInput
    checkIfCommentLine input = check (dropNonNewlineSpace input)
      where
        check input = do
          ('-', input) <- alexGetChar' input
          ('-', input) <- alexGetChar' input
          (c, after_c) <- alexGetChar' input
          case c of
            '-' -> Nothing
            ' ' -> case alexGetChar' after_c of
                     Just ('$', _) -> Nothing
                     _ -> Just input
            _   -> Just input

        dropNonNewlineSpace input = case alexGetChar' input of
          Just (c, input')
            | isSpace c && c /= '\n' -> dropNonNewlineSpace input'
            | otherwise -> input
          Nothing -> input

lineCommentToken :: Action
lineCommentToken span buf len = do
  b <- getBit RawTokenStreamBit
  if b then strtoken ITlineComment span buf len else lexToken

{-
  nested comments require traversing by hand, they can't be parsed
  using regular expressions.
-}
nested_comment :: P (PsLocated Token) -> Action
nested_comment cont span buf len = do
  input <- getInput
  go (reverse $ lexemeToString buf len) (1::Int) input
  where
    go commentAcc 0 input = do
      setInput input
      b <- getBit RawTokenStreamBit
      if b
        then docCommentEnd input commentAcc ITblockComment buf span
        else cont
    go commentAcc n input = case alexGetChar' input of
      Nothing -> errBrace input (psRealSpan span)
      Just ('-',input) -> case alexGetChar' input of
        Nothing  -> errBrace input (psRealSpan span)
        Just ('\125',input) -> go ('\125':'-':commentAcc) (n-1) input -- '}'
        Just (_,_)          -> go ('-':commentAcc) n input
      Just ('\123',input) -> case alexGetChar' input of  -- '{' char
        Nothing  -> errBrace input (psRealSpan span)
        Just ('-',input) -> go ('-':'\123':commentAcc) (n+1) input
        Just (_,_)       -> go ('\123':commentAcc) n input
      -- See Note [Nested comment line pragmas]
      Just ('\n',input) -> case alexGetChar' input of
        Nothing  -> errBrace input (psRealSpan span)
        Just ('#',_) -> do (parsedAcc,input) <- parseNestedPragma input
                           go (parsedAcc ++ '\n':commentAcc) n input
        Just (_,_)   -> go ('\n':commentAcc) n input
      Just (c,input) -> go (c:commentAcc) n input

nested_doc_comment :: Action
nested_doc_comment span buf _len = withLexedDocType (go "")
  where
    go commentAcc input docType _ = case alexGetChar' input of
      Nothing -> errBrace input (psRealSpan span)
      Just ('-',input) -> case alexGetChar' input of
        Nothing -> errBrace input (psRealSpan span)
        Just ('\125',input) ->
          docCommentEnd input commentAcc docType buf span
        Just (_,_) -> go ('-':commentAcc) input docType False
      Just ('\123', input) -> case alexGetChar' input of
        Nothing  -> errBrace input (psRealSpan span)
        Just ('-',input) -> do
          setInput input
          let cont = do input <- getInput; go commentAcc input docType False
          nested_comment cont span buf _len
        Just (_,_) -> go ('\123':commentAcc) input docType False
      -- See Note [Nested comment line pragmas]
      Just ('\n',input) -> case alexGetChar' input of
        Nothing  -> errBrace input (psRealSpan span)
        Just ('#',_) -> do (parsedAcc,input) <- parseNestedPragma input
                           go (parsedAcc ++ '\n':commentAcc) input docType False
        Just (_,_)   -> go ('\n':commentAcc) input docType False
      Just (c,input) -> go (c:commentAcc) input docType False

-- See Note [Nested comment line pragmas]
parseNestedPragma :: AlexInput -> P (String,AlexInput)
parseNestedPragma input@(AI _ buf) = do
  origInput <- getInput
  setInput input
  setExts (.|. xbit InNestedCommentBit)
  pushLexState bol
  lt <- lexToken
  _ <- popLexState
  setExts (.&. complement (xbit InNestedCommentBit))
  postInput@(AI _ postBuf) <- getInput
  setInput origInput
  case unLoc lt of
    ITcomment_line_prag -> do
      let bytes = byteDiff buf postBuf
          diff  = lexemeToString buf bytes
      return (reverse diff, postInput)
    lt' -> panic ("parseNestedPragma: unexpected token" ++ (show lt'))

{-
Note [Nested comment line pragmas]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We used to ignore cpp-preprocessor-generated #line pragmas if they were inside
nested comments.

Now, when parsing a nested comment, if we encounter a line starting with '#' we
call parseNestedPragma, which executes the following:
1. Save the current lexer input (loc, buf) for later
2. Set the current lexer input to the beginning of the line starting with '#'
3. Turn the 'InNestedComment' extension on
4. Push the 'bol' lexer state
5. Lex a token. Due to (2), (3), and (4), this should always lex a single line
   or less and return the ITcomment_line_prag token. This may set source line
   and file location if a #line pragma is successfully parsed
6. Restore lexer input and state to what they were before we did all this
7. Return control to the function parsing a nested comment, informing it of
   what the lexer parsed

Regarding (5) above:
Every exit from the 'bol' lexer state (do_bol, popLinePrag1, failLinePrag1)
checks if the 'InNestedComment' extension is set. If it is, that function will
return control to parseNestedPragma by returning the ITcomment_line_prag token.

See #314 for more background on the bug this fixes.
-}

withLexedDocType :: (AlexInput -> (String -> Token) -> Bool -> P (PsLocated Token))
                 -> P (PsLocated Token)
withLexedDocType lexDocComment = do
  input@(AI _ buf) <- getInput
  case prevChar buf ' ' of
    -- The `Bool` argument to lexDocComment signals whether or not the next
    -- line of input might also belong to this doc comment.
    '|' -> lexDocComment input ITdocCommentNext True
    '^' -> lexDocComment input ITdocCommentPrev True
    '$' -> lexDocComment input ITdocCommentNamed True
    '*' -> lexDocSection 1 input
    _ -> panic "withLexedDocType: Bad doc type"
 where
    lexDocSection n input = case alexGetChar' input of
      Just ('*', input) -> lexDocSection (n+1) input
      Just (_,   _)     -> lexDocComment input (ITdocSection n) False
      Nothing -> do setInput input; lexToken -- eof reached, lex it normally

-- RULES pragmas turn on the forall and '.' keywords, and we turn them
-- off again at the end of the pragma.
rulePrag :: Action
rulePrag span buf len = do
  setExts (.|. xbit InRulePragBit)
  let !src = lexemeToString buf len
  return (L span (ITrules_prag (SourceText src)))

-- When 'UsePosPragsBit' is not set, it is expected that we emit a token instead
-- of updating the position in 'PState'
linePrag :: Action
linePrag span buf len = do
  usePosPrags <- getBit UsePosPragsBit
  if usePosPrags
    then begin line_prag2 span buf len
    else let !src = lexemeToString buf len
         in return (L span (ITline_prag (SourceText src)))

-- When 'UsePosPragsBit' is not set, it is expected that we emit a token instead
-- of updating the position in 'PState'
columnPrag :: Action
columnPrag span buf len = do
  usePosPrags <- getBit UsePosPragsBit
  let !src = lexemeToString buf len
  if usePosPrags
    then begin column_prag span buf len
    else let !src = lexemeToString buf len
         in return (L span (ITcolumn_prag (SourceText src)))

endPrag :: Action
endPrag span _buf _len = do
  setExts (.&. complement (xbit InRulePragBit))
  return (L span ITclose_prag)

-- docCommentEnd
-------------------------------------------------------------------------------
-- This function is quite tricky. We can't just return a new token, we also
-- need to update the state of the parser. Why? Because the token is longer
-- than what was lexed by Alex, and the lexToken function doesn't know this, so
-- it writes the wrong token length to the parser state. This function is
-- called afterwards, so it can just update the state.

docCommentEnd :: AlexInput -> String -> (String -> Token) -> StringBuffer ->
                 PsSpan -> P (PsLocated Token)
docCommentEnd input commentAcc docType buf span = do
  setInput input
  let (AI loc nextBuf) = input
      comment = reverse commentAcc
      span' = mkPsSpan (psSpanStart span) loc
      last_len = byteDiff buf nextBuf

  span `seq` setLastToken span' last_len
  return (L span' (docType comment))

errBrace :: AlexInput -> RealSrcSpan -> P a
errBrace (AI end _) span = failLocMsgP (realSrcSpanStart span) (psRealLoc end) "unterminated `{-'"

open_brace, close_brace :: Action
open_brace span _str _len = do
  ctx <- getContext
  setContext (NoLayout:ctx)
  return (L span ITocurly)
close_brace span _str _len = do
  popContext
  return (L span ITccurly)

qvarid, qconid :: StringBuffer -> Int -> Token
qvarid buf len = ITqvarid $! splitQualName buf len False
qconid buf len = ITqconid $! splitQualName buf len False

splitQualName :: StringBuffer -> Int -> Bool -> (FastString,FastString)
-- takes a StringBuffer and a length, and returns the module name
-- and identifier parts of a qualified name.  Splits at the *last* dot,
-- because of hierarchical module names.
splitQualName orig_buf len parens = split orig_buf orig_buf
  where
    split buf dot_buf
        | orig_buf `byteDiff` buf >= len  = done dot_buf
        | c == '.'                        = found_dot buf'
        | otherwise                       = split buf' dot_buf
      where
       (c,buf') = nextChar buf

    -- careful, we might get names like M....
    -- so, if the character after the dot is not upper-case, this is
    -- the end of the qualifier part.
    found_dot buf -- buf points after the '.'
        | isUpper c    = split buf' buf
        | otherwise    = done buf
      where
       (c,buf') = nextChar buf

    done dot_buf =
        (lexemeToFastString orig_buf (qual_size - 1),
         if parens -- Prelude.(+)
            then lexemeToFastString (stepOn dot_buf) (len - qual_size - 2)
            else lexemeToFastString dot_buf (len - qual_size))
      where
        qual_size = orig_buf `byteDiff` dot_buf

varid :: Action
varid span buf len =
  case lookupUFM reservedWordsFM fs of
    Just (ITcase, _) -> do
      lastTk <- getLastTk
      keyword <- case lastTk of
        Just ITlam -> do
          lambdaCase <- getBit LambdaCaseBit
          unless lambdaCase $ do
            pState <- getPState
            addError (mkSrcSpanPs (last_loc pState)) $ text
                     "Illegal lambda-case (use LambdaCase)"
          return ITlcase
        _ -> return ITcase
      maybe_layout keyword
      return $ L span keyword
    Just (keyword, 0) -> do
      maybe_layout keyword
      return $ L span keyword
    Just (keyword, i) -> do
      exts <- getExts
      if exts .&. i /= 0
        then do
          maybe_layout keyword
          return $ L span keyword
        else
          return $ L span $ ITvarid fs
    Nothing ->
      return $ L span $ ITvarid fs
  where
    !fs = lexemeToFastString buf len

conid :: StringBuffer -> Int -> Token
conid buf len = ITconid $! lexemeToFastString buf len

qvarsym, qconsym :: StringBuffer -> Int -> Token
qvarsym buf len = ITqvarsym $! splitQualName buf len False
qconsym buf len = ITqconsym $! splitQualName buf len False

-- See Note [Whitespace-sensitive operator parsing]
varsym_prefix :: Action
varsym_prefix = sym $ \exts s ->
  if | s == fsLit "@"  -- regardless of TypeApplications for better error messages
     -> return ITtypeApp
     | ThQuotesBit `xtest` exts, s == fsLit "$"
     -> return ITdollar
     | ThQuotesBit `xtest` exts, s == fsLit "$$"
     -> return ITdollardollar
     | s == fsLit "-"   -- Only when LexicalNegation is on, otherwise we get ITminus and
                        -- don't hit this code path. See Note [Minus tokens]
     -> return ITprefixminus
     | s == fsLit "!" -> return ITbang
     | s == fsLit "~" -> return ITtilde
     | otherwise -> return (ITvarsym s)

-- See Note [Whitespace-sensitive operator parsing]
varsym_suffix :: Action
varsym_suffix = sym $ \_ s ->
  if | s == fsLit "@"
     -> failMsgP "Suffix occurrence of @. For an as-pattern, remove the leading whitespace."
     | otherwise -> return (ITvarsym s)

-- See Note [Whitespace-sensitive operator parsing]
varsym_tight_infix :: Action
varsym_tight_infix = sym $ \_ s ->
  if | s == fsLit "@" -> return ITat
     | otherwise -> return (ITvarsym s)

-- See Note [Whitespace-sensitive operator parsing]
varsym_loose_infix :: Action
varsym_loose_infix = sym (\_ s -> return $ ITvarsym s)

consym :: Action
consym = sym (\_exts s -> return $ ITconsym s)

sym :: (ExtsBitmap -> FastString -> P Token) -> Action
sym con span buf len =
  case lookupUFM reservedSymsFM fs of
    Just (keyword, NormalSyntax, 0) ->
      return $ L span keyword
    Just (keyword, NormalSyntax, i) -> do
      exts <- getExts
      if exts .&. i /= 0
        then return $ L span keyword
        else L span <$!> con exts fs
    Just (keyword, UnicodeSyntax, 0) -> do
      exts <- getExts
      if xtest UnicodeSyntaxBit exts
        then return $ L span keyword
        else L span <$!> con exts fs
    Just (keyword, UnicodeSyntax, i) -> do
      exts <- getExts
      if exts .&. i /= 0 && xtest UnicodeSyntaxBit exts
        then return $ L span keyword
        else L span <$!> con exts fs
    Nothing -> do
      exts <- getExts
      L span <$!> con exts fs
  where
    !fs = lexemeToFastString buf len

-- Variations on the integral numeric literal.
tok_integral :: (SourceText -> Integer -> Token)
             -> (Integer -> Integer)
             -> Int -> Int
             -> (Integer, (Char -> Int))
             -> Action
tok_integral itint transint transbuf translen (radix,char_to_int) span buf len = do
  numericUnderscores <- getBit NumericUnderscoresBit  -- #14473
  let src = lexemeToString buf len
  when ((not numericUnderscores) && ('_' `elem` src)) $ do
    pState <- getPState
    addError (mkSrcSpanPs (last_loc pState)) $ text
             "Use NumericUnderscores to allow underscores in integer literals"
  return $ L span $ itint (SourceText src)
       $! transint $ parseUnsignedInteger
       (offsetBytes transbuf buf) (subtract translen len) radix char_to_int

tok_num :: (Integer -> Integer)
        -> Int -> Int
        -> (Integer, (Char->Int)) -> Action
tok_num = tok_integral $ \case
    st@(SourceText ('-':_)) -> itint st (const True)
    st@(SourceText _)       -> itint st (const False)
    st@NoSourceText         -> itint st (< 0)
  where
    itint :: SourceText -> (Integer -> Bool) -> Integer -> Token
    itint !st is_negative !val = ITinteger ((IL st $! is_negative val) val)

tok_primint :: (Integer -> Integer)
            -> Int -> Int
            -> (Integer, (Char->Int)) -> Action
tok_primint = tok_integral ITprimint


tok_primword :: Int -> Int
             -> (Integer, (Char->Int)) -> Action
tok_primword = tok_integral ITprimword positive
positive, negative :: (Integer -> Integer)
positive = id
negative = negate
decimal, octal, hexadecimal :: (Integer, Char -> Int)
decimal = (10,octDecDigit)
binary = (2,octDecDigit)
octal = (8,octDecDigit)
hexadecimal = (16,hexDigit)

-- readRational can understand negative rationals, exponents, everything.
tok_frac :: Int -> (String -> Token) -> Action
tok_frac drop f span buf len = do
  numericUnderscores <- getBit NumericUnderscoresBit  -- #14473
  let src = lexemeToString buf (len-drop)
  when ((not numericUnderscores) && ('_' `elem` src)) $ do
    pState <- getPState
    addError (mkSrcSpanPs (last_loc pState)) $ text
             "Use NumericUnderscores to allow underscores in floating literals"
  return (L span $! (f $! src))

tok_float, tok_primfloat, tok_primdouble :: String -> Token
tok_float        str = ITrational   $! readFractionalLit str
tok_hex_float    str = ITrational   $! readHexFractionalLit str
tok_primfloat    str = ITprimfloat  $! readFractionalLit str
tok_primdouble   str = ITprimdouble $! readFractionalLit str

readFractionalLit :: String -> FractionalLit
readFractionalLit str = ((FL $! (SourceText str)) $! is_neg) $! readRational str
                        where is_neg = case str of ('-':_) -> True
                                                   _       -> False
readHexFractionalLit :: String -> FractionalLit
readHexFractionalLit str =
  FL { fl_text  = SourceText str
     , fl_neg   = case str of
                    '-' : _ -> True
                    _       -> False
     , fl_value = readHexRational str
     }

-- -----------------------------------------------------------------------------
-- Layout processing

-- we're at the first token on a line, insert layout tokens if necessary
do_bol :: Action
do_bol span _str _len = do
        -- See Note [Nested comment line pragmas]
        b <- getBit InNestedCommentBit
        if b then return (L span ITcomment_line_prag) else do
          (pos, gen_semic) <- getOffside
          case pos of
              LT -> do
                  --trace "layout: inserting '}'" $ do
                  popContext
                  -- do NOT pop the lex state, we might have a ';' to insert
                  return (L span ITvccurly)
              EQ | gen_semic -> do
                  --trace "layout: inserting ';'" $ do
                  _ <- popLexState
                  return (L span ITsemi)
              _ -> do
                  _ <- popLexState
                  lexToken

-- certain keywords put us in the "layout" state, where we might
-- add an opening curly brace.
maybe_layout :: Token -> P ()
maybe_layout t = do -- If the alternative layout rule is enabled then
                    -- we never create an implicit layout context here.
                    -- Layout is handled XXX instead.
                    -- The code for closing implicit contexts, or
                    -- inserting implicit semi-colons, is therefore
                    -- irrelevant as it only applies in an implicit
                    -- context.
                    alr <- getBit AlternativeLayoutRuleBit
                    unless alr $ f t
    where f ITdo    = pushLexState layout_do
          f ITmdo   = pushLexState layout_do
          f ITof    = pushLexState layout
          f ITlcase = pushLexState layout
          f ITlet   = pushLexState layout
          f ITwhere = pushLexState layout
          f ITrec   = pushLexState layout
          f ITif    = pushLexState layout_if
          f _       = return ()

-- Pushing a new implicit layout context.  If the indentation of the
-- next token is not greater than the previous layout context, then
-- Haskell 98 says that the new layout context should be empty; that is
-- the lexer must generate {}.
--
-- We are slightly more lenient than this: when the new context is started
-- by a 'do', then we allow the new context to be at the same indentation as
-- the previous context.  This is what the 'strict' argument is for.
new_layout_context :: Bool -> Bool -> Token -> Action
new_layout_context strict gen_semic tok span _buf len = do
    _ <- popLexState
    (AI l _) <- getInput
    let offset = srcLocCol (psRealLoc l) - len
    ctx <- getContext
    nondecreasing <- getBit NondecreasingIndentationBit
    let strict' = strict || not nondecreasing
    case ctx of
        Layout prev_off _ : _  |
           (strict'     && prev_off >= offset  ||
            not strict' && prev_off > offset) -> do
                -- token is indented to the left of the previous context.
                -- we must generate a {} sequence now.
                pushLexState layout_left
                return (L span tok)
        _ -> do setContext (Layout offset gen_semic : ctx)
                return (L span tok)

do_layout_left :: Action
do_layout_left span _buf _len = do
    _ <- popLexState
    pushLexState bol  -- we must be at the start of a line
    return (L span ITvccurly)

-- -----------------------------------------------------------------------------
-- LINE pragmas

setLineAndFile :: Int -> Action
setLineAndFile code (PsSpan span _) buf len = do
  let src = lexemeToString buf (len - 1)  -- drop trailing quotation mark
      linenumLen = length $ head $ words src
      linenum = parseUnsignedInteger buf linenumLen 10 octDecDigit
      file = mkFastString $ go $ drop 1 $ dropWhile (/= '"') src
          -- skip everything through first quotation mark to get to the filename
        where go ('\\':c:cs) = c : go cs
              go (c:cs)      = c : go cs
              go []          = []
              -- decode escapes in the filename.  e.g. on Windows
              -- when our filenames have backslashes in, gcc seems to
              -- escape the backslashes.  One symptom of not doing this
              -- is that filenames in error messages look a bit strange:
              --   C:\\foo\bar.hs
              -- only the first backslash is doubled, because we apply
              -- System.FilePath.normalise before printing out
              -- filenames and it does not remove duplicate
              -- backslashes after the drive letter (should it?).
  resetAlrLastLoc file
  setSrcLoc (mkRealSrcLoc file (fromIntegral linenum - 1) (srcSpanEndCol span))
      -- subtract one: the line number refers to the *following* line
  addSrcFile file
  _ <- popLexState
  pushLexState code
  lexToken

setColumn :: Action
setColumn (PsSpan span _) buf len = do
  let column =
        case reads (lexemeToString buf len) of
          [(column, _)] -> column
          _ -> error "setColumn: expected integer" -- shouldn't happen
  setSrcLoc (mkRealSrcLoc (srcSpanFile span) (srcSpanEndLine span)
                          (fromIntegral (column :: Integer)))
  _ <- popLexState
  lexToken

alrInitialLoc :: FastString -> RealSrcSpan
alrInitialLoc file = mkRealSrcSpan loc loc
    where -- This is a hack to ensure that the first line in a file
          -- looks like it is after the initial location:
          loc = mkRealSrcLoc file (-1) (-1)

-- -----------------------------------------------------------------------------
-- Options, includes and language pragmas.

lex_string_prag :: (String -> Token) -> Action
lex_string_prag mkTok span _buf _len
    = do input <- getInput
         start <- getParsedLoc
         tok <- go [] input
         end <- getParsedLoc
         return (L (mkPsSpan start end) tok)
    where go acc input
              = if isString input "#-}"
                   then do setInput input
                           return (mkTok (reverse acc))
                   else case alexGetChar input of
                          Just (c,i) -> go (c:acc) i
                          Nothing -> err input
          isString _ [] = True
          isString i (x:xs)
              = case alexGetChar i of
                  Just (c,i') | c == x    -> isString i' xs
                  _other -> False
          err (AI end _) = failLocMsgP (realSrcSpanStart (psRealSpan span)) (psRealLoc end) "unterminated options pragma"


-- -----------------------------------------------------------------------------
-- Strings & Chars

-- This stuff is horrible.  I hates it.

lex_string_tok :: Action
lex_string_tok span buf _len = do
  tok <- lex_string ""
  (AI end bufEnd) <- getInput
  let
    tok' = case tok of
            ITprimstring _ bs -> ITprimstring (SourceText src) bs
            ITstring _ s -> ITstring (SourceText src) s
            _ -> panic "lex_string_tok"
    src = lexemeToString buf (cur bufEnd - cur buf)
  return (L (mkPsSpan (psSpanStart span) end) tok')

lex_string :: String -> P Token
lex_string s = do
  i <- getInput
  case alexGetChar' i of
    Nothing -> lit_error i

    Just ('"',i)  -> do
        setInput i
        let s' = reverse s
        magicHash <- getBit MagicHashBit
        if magicHash
          then do
            i <- getInput
            case alexGetChar' i of
              Just ('#',i) -> do
                setInput i
                when (any (> '\xFF') s') $ do
                  pState <- getPState
                  addError (mkSrcSpanPs (last_loc pState)) $ text
                     "primitive string literal must contain only characters <= \'\\xFF\'"
                return (ITprimstring (SourceText s') (unsafeMkByteString s'))
              _other ->
                return (ITstring (SourceText s') (mkFastString s'))
          else
                return (ITstring (SourceText s') (mkFastString s'))

    Just ('\\',i)
        | Just ('&',i) <- next -> do
                setInput i; lex_string s
        | Just (c,i) <- next, c <= '\x7f' && is_space c -> do
                           -- is_space only works for <= '\x7f' (#3751, #5425)
                setInput i; lex_stringgap s
        where next = alexGetChar' i

    Just (c, i1) -> do
        case c of
          '\\' -> do setInput i1; c' <- lex_escape; lex_string (c':s)
          c | isAny c -> do setInput i1; lex_string (c:s)
          _other -> lit_error i

lex_stringgap :: String -> P Token
lex_stringgap s = do
  i <- getInput
  c <- getCharOrFail i
  case c of
    '\\' -> lex_string s
    c | c <= '\x7f' && is_space c -> lex_stringgap s
                           -- is_space only works for <= '\x7f' (#3751, #5425)
    _other -> lit_error i


lex_char_tok :: Action
-- Here we are basically parsing character literals, such as 'x' or '\n'
-- but we additionally spot 'x and ''T, returning ITsimpleQuote and
-- ITtyQuote respectively, but WITHOUT CONSUMING the x or T part
-- (the parser does that).
-- So we have to do two characters of lookahead: when we see 'x we need to
-- see if there's a trailing quote
lex_char_tok span buf _len = do        -- We've seen '
   i1 <- getInput       -- Look ahead to first character
   let loc = psSpanStart span
   case alexGetChar' i1 of
        Nothing -> lit_error  i1

        Just ('\'', i2@(AI end2 _)) -> do       -- We've seen ''
                   setInput i2
                   return (L (mkPsSpan loc end2)  ITtyQuote)

        Just ('\\', i2@(AI _end2 _)) -> do      -- We've seen 'backslash
                  setInput i2
                  lit_ch <- lex_escape
                  i3 <- getInput
                  mc <- getCharOrFail i3 -- Trailing quote
                  if mc == '\'' then finish_char_tok buf loc lit_ch
                                else lit_error i3

        Just (c, i2@(AI _end2 _))
                | not (isAny c) -> lit_error i1
                | otherwise ->

                -- We've seen 'x, where x is a valid character
                --  (i.e. not newline etc) but not a quote or backslash
           case alexGetChar' i2 of      -- Look ahead one more character
                Just ('\'', i3) -> do   -- We've seen 'x'
                        setInput i3
                        finish_char_tok buf loc c
                _other -> do            -- We've seen 'x not followed by quote
                                        -- (including the possibility of EOF)
                                        -- Just parse the quote only
                        let (AI end _) = i1
                        return (L (mkPsSpan loc end) ITsimpleQuote)

finish_char_tok :: StringBuffer -> PsLoc -> Char -> P (PsLocated Token)
finish_char_tok buf loc ch  -- We've already seen the closing quote
                        -- Just need to check for trailing #
  = do  magicHash <- getBit MagicHashBit
        i@(AI end bufEnd) <- getInput
        let src = lexemeToString buf (cur bufEnd - cur buf)
        if magicHash then do
            case alexGetChar' i of
              Just ('#',i@(AI end _)) -> do
                setInput i
                return (L (mkPsSpan loc end)
                          (ITprimchar (SourceText src) ch))
              _other ->
                return (L (mkPsSpan loc end)
                          (ITchar (SourceText src) ch))
            else do
              return (L (mkPsSpan loc end) (ITchar (SourceText src) ch))

isAny :: Char -> Bool
isAny c | c > '\x7f' = isPrint c
        | otherwise  = is_any c

lex_escape :: P Char
lex_escape = do
  i0 <- getInput
  c <- getCharOrFail i0
  case c of
        'a'   -> return '\a'
        'b'   -> return '\b'
        'f'   -> return '\f'
        'n'   -> return '\n'
        'r'   -> return '\r'
        't'   -> return '\t'
        'v'   -> return '\v'
        '\\'  -> return '\\'
        '"'   -> return '\"'
        '\''  -> return '\''
        '^'   -> do i1 <- getInput
                    c <- getCharOrFail i1
                    if c >= '@' && c <= '_'
                        then return (chr (ord c - ord '@'))
                        else lit_error i1

        'x'   -> readNum is_hexdigit 16 hexDigit
        'o'   -> readNum is_octdigit  8 octDecDigit
        x | is_decdigit x -> readNum2 is_decdigit 10 octDecDigit (octDecDigit x)

        c1 ->  do
           i <- getInput
           case alexGetChar' i of
            Nothing -> lit_error i0
            Just (c2,i2) ->
              case alexGetChar' i2 of
                Nothing -> do lit_error i0
                Just (c3,i3) ->
                   let str = [c1,c2,c3] in
                   case [ (c,rest) | (p,c) <- silly_escape_chars,
                                     Just rest <- [stripPrefix p str] ] of
                          (escape_char,[]):_ -> do
                                setInput i3
                                return escape_char
                          (escape_char,_:_):_ -> do
                                setInput i2
                                return escape_char
                          [] -> lit_error i0

readNum :: (Char -> Bool) -> Int -> (Char -> Int) -> P Char
readNum is_digit base conv = do
  i <- getInput
  c <- getCharOrFail i
  if is_digit c
        then readNum2 is_digit base conv (conv c)
        else lit_error i

readNum2 :: (Char -> Bool) -> Int -> (Char -> Int) -> Int -> P Char
readNum2 is_digit base conv i = do
  input <- getInput
  read i input
  where read i input = do
          case alexGetChar' input of
            Just (c,input') | is_digit c -> do
               let i' = i*base + conv c
               if i' > 0x10ffff
                  then setInput input >> lexError "numeric escape sequence out of range"
                  else read i' input'
            _other -> do
              setInput input; return (chr i)


silly_escape_chars :: [(String, Char)]
silly_escape_chars = [
        ("NUL", '\NUL'),
        ("SOH", '\SOH'),
        ("STX", '\STX'),
        ("ETX", '\ETX'),
        ("EOT", '\EOT'),
        ("ENQ", '\ENQ'),
        ("ACK", '\ACK'),
        ("BEL", '\BEL'),
        ("BS", '\BS'),
        ("HT", '\HT'),
        ("LF", '\LF'),
        ("VT", '\VT'),
        ("FF", '\FF'),
        ("CR", '\CR'),
        ("SO", '\SO'),
        ("SI", '\SI'),
        ("DLE", '\DLE'),
        ("DC1", '\DC1'),
        ("DC2", '\DC2'),
        ("DC3", '\DC3'),
        ("DC4", '\DC4'),
        ("NAK", '\NAK'),
        ("SYN", '\SYN'),
        ("ETB", '\ETB'),
        ("CAN", '\CAN'),
        ("EM", '\EM'),
        ("SUB", '\SUB'),
        ("ESC", '\ESC'),
        ("FS", '\FS'),
        ("GS", '\GS'),
        ("RS", '\RS'),
        ("US", '\US'),
        ("SP", '\SP'),
        ("DEL", '\DEL')
        ]

-- before calling lit_error, ensure that the current input is pointing to
-- the position of the error in the buffer.  This is so that we can report
-- a correct location to the user, but also so we can detect UTF-8 decoding
-- errors if they occur.
lit_error :: AlexInput -> P a
lit_error i = do setInput i; lexError "lexical error in string/character literal"

getCharOrFail :: AlexInput -> P Char
getCharOrFail i =  do
  case alexGetChar' i of
        Nothing -> lexError "unexpected end-of-file in string/character literal"
        Just (c,i)  -> do setInput i; return c

-- -----------------------------------------------------------------------------
-- QuasiQuote

lex_qquasiquote_tok :: Action
lex_qquasiquote_tok span buf len = do
  let (qual, quoter) = splitQualName (stepOn buf) (len - 2) False
  quoteStart <- getParsedLoc
  quote <- lex_quasiquote (psRealLoc quoteStart) ""
  end <- getParsedLoc
  return (L (mkPsSpan (psSpanStart span) end)
           (ITqQuasiQuote (qual,
                           quoter,
                           mkFastString (reverse quote),
                           mkPsSpan quoteStart end)))

lex_quasiquote_tok :: Action
lex_quasiquote_tok span buf len = do
  let quoter = tail (lexemeToString buf (len - 1))
                -- 'tail' drops the initial '[',
                -- while the -1 drops the trailing '|'
  quoteStart <- getParsedLoc
  quote <- lex_quasiquote (psRealLoc quoteStart) ""
  end <- getParsedLoc
  return (L (mkPsSpan (psSpanStart span) end)
           (ITquasiQuote (mkFastString quoter,
                          mkFastString (reverse quote),
                          mkPsSpan quoteStart end)))

lex_quasiquote :: RealSrcLoc -> String -> P String
lex_quasiquote start s = do
  i <- getInput
  case alexGetChar' i of
    Nothing -> quasiquote_error start

    -- NB: The string "|]" terminates the quasiquote,
    -- with absolutely no escaping. See the extensive
    -- discussion on #5348 for why there is no
    -- escape handling.
    Just ('|',i)
        | Just (']',i) <- alexGetChar' i
        -> do { setInput i; return s }

    Just (c, i) -> do
         setInput i; lex_quasiquote start (c : s)

quasiquote_error :: RealSrcLoc -> P a
quasiquote_error start = do
  (AI end buf) <- getInput
  reportLexError start (psRealLoc end) buf "unterminated quasiquotation"

-- -----------------------------------------------------------------------------
-- Warnings

warnTab :: Action
warnTab srcspan _buf _len = do
    addTabWarning (psRealSpan srcspan)
    lexToken

warnThen :: WarningFlag -> SDoc -> Action -> Action
warnThen option warning action srcspan buf len = do
    addWarning option (RealSrcSpan (psRealSpan srcspan) Nothing) warning
    action srcspan buf len

-- -----------------------------------------------------------------------------
-- The Parse Monad

-- | Do we want to generate ';' layout tokens? In some cases we just want to
-- generate '}', e.g. in MultiWayIf we don't need ';'s because '|' separates
-- alternatives (unlike a `case` expression where we need ';' to as a separator
-- between alternatives).
type GenSemic = Bool

generateSemic, dontGenerateSemic :: GenSemic
generateSemic     = True
dontGenerateSemic = False

data LayoutContext
  = NoLayout
  | Layout !Int !GenSemic
  deriving Show

-- | The result of running a parser.
data ParseResult a
  = POk      -- ^ The parser has consumed a (possibly empty) prefix
             --   of the input and produced a result. Use 'getMessages'
             --   to check for accumulated warnings and non-fatal errors.
      PState -- ^ The resulting parsing state. Can be used to resume parsing.
      a      -- ^ The resulting value.
  | PFailed  -- ^ The parser has consumed a (possibly empty) prefix
             --   of the input and failed.
      PState -- ^ The parsing state right before failure, including the fatal
             --   parse error. 'getMessages' and 'getErrorMessages' must return
             --   a non-empty bag of errors.

-- | Test whether a 'WarningFlag' is set
warnopt :: WarningFlag -> ParserFlags -> Bool
warnopt f options = f `EnumSet.member` pWarningFlags options

-- | The subset of the 'DynFlags' used by the parser.
-- See 'mkParserFlags' or 'mkParserFlags'' for ways to construct this.
data ParserFlags = ParserFlags {
    pWarningFlags   :: EnumSet WarningFlag
  , pHomeUnitId     :: UnitId      -- ^ unit currently being compiled
  , pExtsBitmap     :: !ExtsBitmap -- ^ bitmap of permitted extensions
  }

data PState = PState {
        buffer     :: StringBuffer,
        options    :: ParserFlags,
        -- This needs to take DynFlags as an argument until
        -- we have a fix for #10143
        messages   :: DynFlags -> Messages,
        tab_first  :: Maybe RealSrcSpan, -- pos of first tab warning in the file
        tab_count  :: !Int,              -- number of tab warnings in the file
        last_tk    :: Maybe Token,
        last_loc   :: PsSpan,      -- pos of previous token
        last_len   :: !Int,        -- len of previous token
        loc        :: PsLoc,       -- current loc (end of prev token + 1)
        context    :: [LayoutContext],
        lex_state  :: [Int],
        srcfiles   :: [FastString],
        -- Used in the alternative layout rule:
        -- These tokens are the next ones to be sent out. They are
        -- just blindly emitted, without the rule looking at them again:
        alr_pending_implicit_tokens :: [PsLocated Token],
        -- This is the next token to be considered or, if it is Nothing,
        -- we need to get the next token from the input stream:
        alr_next_token :: Maybe (PsLocated Token),
        -- This is what we consider to be the location of the last token
        -- emitted:
        alr_last_loc :: PsSpan,
        -- The stack of layout contexts:
        alr_context :: [ALRContext],
        -- Are we expecting a '{'? If it's Just, then the ALRLayout tells
        -- us what sort of layout the '{' will open:
        alr_expecting_ocurly :: Maybe ALRLayout,
        -- Have we just had the '}' for a let block? If so, than an 'in'
        -- token doesn't need to close anything:
        alr_justClosedExplicitLetBlock :: Bool,

        -- The next three are used to implement Annotations giving the
        -- locations of 'noise' tokens in the source, so that users of
        -- the GHC API can do source to source conversions.
        -- See note [Api annotations] in GHC.Parser.Annotation
        annotations :: [(ApiAnnKey,[RealSrcSpan])],
        eof_pos :: Maybe RealSrcSpan,
        comment_q :: [RealLocated AnnotationComment],
        annotations_comments :: [(RealSrcSpan,[RealLocated AnnotationComment])]
     }
        -- last_loc and last_len are used when generating error messages,
        -- and in pushCurrentContext only.  Sigh, if only Happy passed the
        -- current token to happyError, we could at least get rid of last_len.
        -- Getting rid of last_loc would require finding another way to
        -- implement pushCurrentContext (which is only called from one place).

data ALRContext = ALRNoLayout Bool{- does it contain commas? -}
                              Bool{- is it a 'let' block? -}
                | ALRLayout ALRLayout Int
data ALRLayout = ALRLayoutLet
               | ALRLayoutWhere
               | ALRLayoutOf
               | ALRLayoutDo

-- | The parsing monad, isomorphic to @StateT PState Maybe@.
newtype P a = P { unP :: PState -> ParseResult a }

instance Functor P where
  fmap = liftM

instance Applicative P where
  pure = returnP
  (<*>) = ap

instance Monad P where
  (>>=) = thenP

returnP :: a -> P a
returnP a = a `seq` (P $ \s -> POk s a)

thenP :: P a -> (a -> P b) -> P b
(P m) `thenP` k = P $ \ s ->
        case m s of
                POk s1 a         -> (unP (k a)) s1
                PFailed s1 -> PFailed s1

failMsgP :: String -> P a
failMsgP msg = do
  pState <- getPState
  addFatalError (mkSrcSpanPs (last_loc pState)) (text msg)

failLocMsgP :: RealSrcLoc -> RealSrcLoc -> String -> P a
failLocMsgP loc1 loc2 str =
  addFatalError (RealSrcSpan (mkRealSrcSpan loc1 loc2) Nothing) (text str)

getPState :: P PState
getPState = P $ \s -> POk s s

withHomeUnitId :: (UnitId -> a) -> P a
withHomeUnitId f = P $ \s@(PState{options = o}) -> POk s (f (pHomeUnitId o))

getExts :: P ExtsBitmap
getExts = P $ \s -> POk s (pExtsBitmap . options $ s)

setExts :: (ExtsBitmap -> ExtsBitmap) -> P ()
setExts f = P $ \s -> POk s {
  options =
    let p = options s
    in  p { pExtsBitmap = f (pExtsBitmap p) }
  } ()

setSrcLoc :: RealSrcLoc -> P ()
setSrcLoc new_loc =
  P $ \s@(PState{ loc = PsLoc _ buf_loc }) ->
  POk s{ loc = PsLoc new_loc buf_loc } ()

getRealSrcLoc :: P RealSrcLoc
getRealSrcLoc = P $ \s@(PState{ loc=loc }) -> POk s (psRealLoc loc)

getParsedLoc :: P PsLoc
getParsedLoc  = P $ \s@(PState{ loc=loc }) -> POk s loc

addSrcFile :: FastString -> P ()
addSrcFile f = P $ \s -> POk s{ srcfiles = f : srcfiles s } ()

setEofPos :: RealSrcSpan -> P ()
setEofPos span = P $ \s -> POk s{ eof_pos = Just span } ()

setLastToken :: PsSpan -> Int -> P ()
setLastToken loc len = P $ \s -> POk s {
  last_loc=loc,
  last_len=len
  } ()

setLastTk :: Token -> P ()
setLastTk tk = P $ \s -> POk s { last_tk = Just tk } ()

getLastTk :: P (Maybe Token)
getLastTk = P $ \s@(PState { last_tk = last_tk }) -> POk s last_tk

data AlexInput = AI PsLoc StringBuffer

{-
Note [Unicode in Alex]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Although newer versions of Alex support unicode, this grammar is processed with
the old style '--latin1' behaviour. This means that when implementing the
functions

    alexGetByte       :: AlexInput -> Maybe (Word8,AlexInput)
    alexInputPrevChar :: AlexInput -> Char

which Alex uses to take apart our 'AlexInput', we must

  * return a latin1 character in the 'Word8' that 'alexGetByte' expects
  * return a latin1 character in 'alexInputPrevChar'.

We handle this in 'adjustChar' by squishing entire classes of unicode
characters into single bytes.
-}

{-# INLINE adjustChar #-}
adjustChar :: Char -> Word8
adjustChar c = fromIntegral $ ord adj_c
  where non_graphic     = '\x00'
        upper           = '\x01'
        lower           = '\x02'
        digit           = '\x03'
        symbol          = '\x04'
        space           = '\x05'
        other_graphic   = '\x06'
        uniidchar       = '\x07'

        adj_c
          | c <= '\x07' = non_graphic
          | c <= '\x7f' = c
          -- Alex doesn't handle Unicode, so when Unicode
          -- character is encountered we output these values
          -- with the actual character value hidden in the state.
          | otherwise =
                -- NB: The logic behind these definitions is also reflected
                -- in "GHC.Utils.Lexeme"
                -- Any changes here should likely be reflected there.

                case generalCategory c of
                  UppercaseLetter       -> upper
                  LowercaseLetter       -> lower
                  TitlecaseLetter       -> upper
                  ModifierLetter        -> uniidchar -- see #10196
                  OtherLetter           -> lower -- see #1103
                  NonSpacingMark        -> uniidchar -- see #7650
                  SpacingCombiningMark  -> other_graphic
                  EnclosingMark         -> other_graphic
                  DecimalNumber         -> digit
                  LetterNumber          -> other_graphic
                  OtherNumber           -> digit -- see #4373
                  ConnectorPunctuation  -> symbol
                  DashPunctuation       -> symbol
                  OpenPunctuation       -> other_graphic
                  ClosePunctuation      -> other_graphic
                  InitialQuote          -> other_graphic
                  FinalQuote            -> other_graphic
                  OtherPunctuation      -> symbol
                  MathSymbol            -> symbol
                  CurrencySymbol        -> symbol
                  ModifierSymbol        -> symbol
                  OtherSymbol           -> symbol
                  Space                 -> space
                  _other                -> non_graphic

-- Getting the previous 'Char' isn't enough here - we need to convert it into
-- the same format that 'alexGetByte' would have produced.
--
-- See Note [Unicode in Alex] and #13986.
alexInputPrevChar :: AlexInput -> Char
alexInputPrevChar (AI _ buf) = chr (fromIntegral (adjustChar pc))
  where pc = prevChar buf '\n'

-- backwards compatibility for Alex 2.x
alexGetChar :: AlexInput -> Maybe (Char,AlexInput)
alexGetChar inp = case alexGetByte inp of
                    Nothing    -> Nothing
                    Just (b,i) -> c `seq` Just (c,i)
                       where c = chr $ fromIntegral b

-- See Note [Unicode in Alex]
alexGetByte :: AlexInput -> Maybe (Word8,AlexInput)
alexGetByte (AI loc s)
  | atEnd s   = Nothing
  | otherwise = byte `seq` loc' `seq` s' `seq`
                --trace (show (ord c)) $
                Just (byte, (AI loc' s'))
  where (c,s') = nextChar s
        loc'   = advancePsLoc loc c
        byte   = adjustChar c

-- This version does not squash unicode characters, it is used when
-- lexing strings.
alexGetChar' :: AlexInput -> Maybe (Char,AlexInput)
alexGetChar' (AI loc s)
  | atEnd s   = Nothing
  | otherwise = c `seq` loc' `seq` s' `seq`
                --trace (show (ord c)) $
                Just (c, (AI loc' s'))
  where (c,s') = nextChar s
        loc'   = advancePsLoc loc c

getInput :: P AlexInput
getInput = P $ \s@PState{ loc=l, buffer=b } -> POk s (AI l b)

setInput :: AlexInput -> P ()
setInput (AI l b) = P $ \s -> POk s{ loc=l, buffer=b } ()

nextIsEOF :: P Bool
nextIsEOF = do
  AI _ s <- getInput
  return $ atEnd s

pushLexState :: Int -> P ()
pushLexState ls = P $ \s@PState{ lex_state=l } -> POk s{lex_state=ls:l} ()

popLexState :: P Int
popLexState = P $ \s@PState{ lex_state=ls:l } -> POk s{ lex_state=l } ls

getLexState :: P Int
getLexState = P $ \s@PState{ lex_state=ls:_ } -> POk s ls

popNextToken :: P (Maybe (PsLocated Token))
popNextToken
    = P $ \s@PState{ alr_next_token = m } ->
              POk (s {alr_next_token = Nothing}) m

activeContext :: P Bool
activeContext = do
  ctxt <- getALRContext
  expc <- getAlrExpectingOCurly
  impt <- implicitTokenPending
  case (ctxt,expc) of
    ([],Nothing) -> return impt
    _other       -> return True

resetAlrLastLoc :: FastString -> P ()
resetAlrLastLoc file =
  P $ \s@(PState {alr_last_loc = PsSpan _ buf_span}) ->
  POk s{ alr_last_loc = PsSpan (alrInitialLoc file) buf_span } ()

setAlrLastLoc :: PsSpan -> P ()
setAlrLastLoc l = P $ \s -> POk (s {alr_last_loc = l}) ()

getAlrLastLoc :: P PsSpan
getAlrLastLoc = P $ \s@(PState {alr_last_loc = l}) -> POk s l

getALRContext :: P [ALRContext]
getALRContext = P $ \s@(PState {alr_context = cs}) -> POk s cs

setALRContext :: [ALRContext] -> P ()
setALRContext cs = P $ \s -> POk (s {alr_context = cs}) ()

getJustClosedExplicitLetBlock :: P Bool
getJustClosedExplicitLetBlock
 = P $ \s@(PState {alr_justClosedExplicitLetBlock = b}) -> POk s b

setJustClosedExplicitLetBlock :: Bool -> P ()
setJustClosedExplicitLetBlock b
 = P $ \s -> POk (s {alr_justClosedExplicitLetBlock = b}) ()

setNextToken :: PsLocated Token -> P ()
setNextToken t = P $ \s -> POk (s {alr_next_token = Just t}) ()

implicitTokenPending :: P Bool
implicitTokenPending
    = P $ \s@PState{ alr_pending_implicit_tokens = ts } ->
              case ts of
              [] -> POk s False
              _  -> POk s True

popPendingImplicitToken :: P (Maybe (PsLocated Token))
popPendingImplicitToken
    = P $ \s@PState{ alr_pending_implicit_tokens = ts } ->
              case ts of
              [] -> POk s Nothing
              (t : ts') -> POk (s {alr_pending_implicit_tokens = ts'}) (Just t)

setPendingImplicitTokens :: [PsLocated Token] -> P ()
setPendingImplicitTokens ts = P $ \s -> POk (s {alr_pending_implicit_tokens = ts}) ()

getAlrExpectingOCurly :: P (Maybe ALRLayout)
getAlrExpectingOCurly = P $ \s@(PState {alr_expecting_ocurly = b}) -> POk s b

setAlrExpectingOCurly :: Maybe ALRLayout -> P ()
setAlrExpectingOCurly b = P $ \s -> POk (s {alr_expecting_ocurly = b}) ()

-- | For reasons of efficiency, boolean parsing flags (eg, language extensions
-- or whether we are currently in a @RULE@ pragma) are represented by a bitmap
-- stored in a @Word64@.
type ExtsBitmap = Word64

xbit :: ExtBits -> ExtsBitmap
xbit = bit . fromEnum

xtest :: ExtBits -> ExtsBitmap -> Bool
xtest ext xmap = testBit xmap (fromEnum ext)

-- | Various boolean flags, mostly language extensions, that impact lexing and
-- parsing. Note that a handful of these can change during lexing/parsing.
data ExtBits
  -- Flags that are constant once parsing starts
  = FfiBit
  | InterruptibleFfiBit
  | CApiFfiBit
  | ArrowsBit
  | ThBit
  | ThQuotesBit
  | IpBit
  | OverloadedLabelsBit -- #x overloaded labels
  | ExplicitForallBit -- the 'forall' keyword
  | BangPatBit -- Tells the parser to understand bang-patterns
               -- (doesn't affect the lexer)
  | PatternSynonymsBit -- pattern synonyms
  | HaddockBit-- Lex and parse Haddock comments
  | MagicHashBit -- "#" in both functions and operators
  | RecursiveDoBit -- mdo
  | UnicodeSyntaxBit -- the forall symbol, arrow symbols, etc
  | UnboxedTuplesBit -- (# and #)
  | UnboxedSumsBit -- (# and #)
  | DatatypeContextsBit
  | MonadComprehensionsBit
  | TransformComprehensionsBit
  | QqBit -- enable quasiquoting
  | RawTokenStreamBit -- producing a token stream with all comments included
  | AlternativeLayoutRuleBit
  | ALRTransitionalBit
  | RelaxedLayoutBit
  | NondecreasingIndentationBit
  | SafeHaskellBit
  | TraditionalRecordSyntaxBit
  | ExplicitNamespacesBit
  | LambdaCaseBit
  | BinaryLiteralsBit
  | NegativeLiteralsBit
  | HexFloatLiteralsBit
  | StaticPointersBit
  | NumericUnderscoresBit
  | StarIsTypeBit
  | BlockArgumentsBit
  | NPlusKPatternsBit
  | DoAndIfThenElseBit
  | MultiWayIfBit
  | GadtSyntaxBit
  | ImportQualifiedPostBit
  | LinearTypesBit
  | NoLexicalNegationBit   -- See Note [Why not LexicalNegationBit]

  -- Flags that are updated once parsing starts
  | InRulePragBit
  | InNestedCommentBit -- See Note [Nested comment line pragmas]
  | UsePosPragsBit
    -- ^ If this is enabled, '{-# LINE ... -#}' and '{-# COLUMN ... #-}'
    -- update the internal position. Otherwise, those pragmas are lexed as
    -- tokens of their own.
  deriving Enum





-- PState for parsing options pragmas
--
pragState :: DynFlags -> StringBuffer -> RealSrcLoc -> PState
pragState dynflags buf loc = (mkPState dynflags buf loc) {
                                 lex_state = [bol, option_prags, 0]
                             }

{-# INLINE mkParserFlags' #-}
mkParserFlags'
  :: EnumSet WarningFlag        -- ^ warnings flags enabled
  -> EnumSet LangExt.Extension  -- ^ permitted language extensions enabled
  -> UnitId                     -- ^ id of the unit currently being compiled
  -> Bool                       -- ^ are safe imports on?
  -> Bool                       -- ^ keeping Haddock comment tokens
  -> Bool                       -- ^ keep regular comment tokens

  -> Bool
  -- ^ If this is enabled, '{-# LINE ... -#}' and '{-# COLUMN ... #-}' update
  -- the internal position kept by the parser. Otherwise, those pragmas are
  -- lexed as 'ITline_prag' and 'ITcolumn_prag' tokens.

  -> ParserFlags
-- ^ Given exactly the information needed, set up the 'ParserFlags'
mkParserFlags' warningFlags extensionFlags homeUnitId
  safeImports isHaddock rawTokStream usePosPrags =
    ParserFlags {
      pWarningFlags = warningFlags
    , pHomeUnitId   = homeUnitId
    , pExtsBitmap   = safeHaskellBit .|. langExtBits .|. optBits
    }
  where
    safeHaskellBit = SafeHaskellBit `setBitIf` safeImports
    langExtBits =
          FfiBit                      `xoptBit` LangExt.ForeignFunctionInterface
      .|. InterruptibleFfiBit         `xoptBit` LangExt.InterruptibleFFI
      .|. CApiFfiBit                  `xoptBit` LangExt.CApiFFI
      .|. ArrowsBit                   `xoptBit` LangExt.Arrows
      .|. ThBit                       `xoptBit` LangExt.TemplateHaskell
      .|. ThQuotesBit                 `xoptBit` LangExt.TemplateHaskellQuotes
      .|. QqBit                       `xoptBit` LangExt.QuasiQuotes
      .|. IpBit                       `xoptBit` LangExt.ImplicitParams
      .|. OverloadedLabelsBit         `xoptBit` LangExt.OverloadedLabels
      .|. ExplicitForallBit           `xoptBit` LangExt.ExplicitForAll
      .|. BangPatBit                  `xoptBit` LangExt.BangPatterns
      .|. MagicHashBit                `xoptBit` LangExt.MagicHash
      .|. RecursiveDoBit              `xoptBit` LangExt.RecursiveDo
      .|. UnicodeSyntaxBit            `xoptBit` LangExt.UnicodeSyntax
      .|. UnboxedTuplesBit            `xoptBit` LangExt.UnboxedTuples
      .|. UnboxedSumsBit              `xoptBit` LangExt.UnboxedSums
      .|. DatatypeContextsBit         `xoptBit` LangExt.DatatypeContexts
      .|. TransformComprehensionsBit  `xoptBit` LangExt.TransformListComp
      .|. MonadComprehensionsBit      `xoptBit` LangExt.MonadComprehensions
      .|. AlternativeLayoutRuleBit    `xoptBit` LangExt.AlternativeLayoutRule
      .|. ALRTransitionalBit          `xoptBit` LangExt.AlternativeLayoutRuleTransitional
      .|. RelaxedLayoutBit            `xoptBit` LangExt.RelaxedLayout
      .|. NondecreasingIndentationBit `xoptBit` LangExt.NondecreasingIndentation
      .|. TraditionalRecordSyntaxBit  `xoptBit` LangExt.TraditionalRecordSyntax
      .|. ExplicitNamespacesBit       `xoptBit` LangExt.ExplicitNamespaces
      .|. LambdaCaseBit               `xoptBit` LangExt.LambdaCase
      .|. BinaryLiteralsBit           `xoptBit` LangExt.BinaryLiterals
      .|. NegativeLiteralsBit         `xoptBit` LangExt.NegativeLiterals
      .|. HexFloatLiteralsBit         `xoptBit` LangExt.HexFloatLiterals
      .|. PatternSynonymsBit          `xoptBit` LangExt.PatternSynonyms
      .|. StaticPointersBit           `xoptBit` LangExt.StaticPointers
      .|. NumericUnderscoresBit       `xoptBit` LangExt.NumericUnderscores
      .|. StarIsTypeBit               `xoptBit` LangExt.StarIsType
      .|. BlockArgumentsBit           `xoptBit` LangExt.BlockArguments
      .|. NPlusKPatternsBit           `xoptBit` LangExt.NPlusKPatterns
      .|. DoAndIfThenElseBit          `xoptBit` LangExt.DoAndIfThenElse
      .|. MultiWayIfBit               `xoptBit` LangExt.MultiWayIf
      .|. GadtSyntaxBit               `xoptBit` LangExt.GADTSyntax
      .|. ImportQualifiedPostBit      `xoptBit` LangExt.ImportQualifiedPost
      .|. LinearTypesBit              `xoptBit` LangExt.LinearTypes
      .|. NoLexicalNegationBit     `xoptNotBit` LangExt.LexicalNegation -- See Note [Why not LexicalNegationBit]
    optBits =
          HaddockBit        `setBitIf` isHaddock
      .|. RawTokenStreamBit `setBitIf` rawTokStream
      .|. UsePosPragsBit    `setBitIf` usePosPrags

    xoptBit bit ext = bit `setBitIf` EnumSet.member ext extensionFlags
    xoptNotBit bit ext = bit `setBitIf` not (EnumSet.member ext extensionFlags)

    setBitIf :: ExtBits -> Bool -> ExtsBitmap
    b `setBitIf` cond | cond      = xbit b
                      | otherwise = 0

-- | Extracts the flag information needed for parsing
mkParserFlags :: DynFlags -> ParserFlags
mkParserFlags =
  mkParserFlags'
    <$> DynFlags.warningFlags
    <*> DynFlags.extensionFlags
    <*> DynFlags.homeUnitId
    <*> safeImportsOn
    <*> gopt Opt_Haddock
    <*> gopt Opt_KeepRawTokenStream
    <*> const True

-- | Creates a parse state from a 'DynFlags' value
mkPState :: DynFlags -> StringBuffer -> RealSrcLoc -> PState
mkPState flags = mkPStatePure (mkParserFlags flags)

-- | Creates a parse state from a 'ParserFlags' value
mkPStatePure :: ParserFlags -> StringBuffer -> RealSrcLoc -> PState
mkPStatePure options buf loc =
  PState {
      buffer        = buf,
      options       = options,
      messages      = const emptyMessages,
      tab_first     = Nothing,
      tab_count     = 0,
      last_tk       = Nothing,
      last_loc      = mkPsSpan init_loc init_loc,
      last_len      = 0,
      loc           = init_loc,
      context       = [],
      lex_state     = [bol, 0],
      srcfiles      = [],
      alr_pending_implicit_tokens = [],
      alr_next_token = Nothing,
      alr_last_loc = PsSpan (alrInitialLoc (fsLit "<no file>")) (BufSpan (BufPos 0) (BufPos 0)),
      alr_context = [],
      alr_expecting_ocurly = Nothing,
      alr_justClosedExplicitLetBlock = False,
      annotations = [],
      eof_pos = Nothing,
      comment_q = [],
      annotations_comments = []
    }
  where init_loc = PsLoc loc (BufPos 0)

-- | An mtl-style class for monads that support parsing-related operations.
-- For example, sometimes we make a second pass over the parsing results to validate,
-- disambiguate, or rearrange them, and we do so in the PV monad which cannot consume
-- input but can report parsing errors, check for extension bits, and accumulate
-- parsing annotations. Both P and PV are instances of MonadP.
--
-- MonadP grants us convenient overloading. The other option is to have separate operations
-- for each monad: addErrorP vs addErrorPV, getBitP vs getBitPV, and so on.
--
class Monad m => MonadP m where
  -- | Add a non-fatal error. Use this when the parser can produce a result
  --   despite the error.
  --
  --   For example, when GHC encounters a @forall@ in a type,
  --   but @-XExplicitForAll@ is disabled, the parser constructs @ForAllTy@
  --   as if @-XExplicitForAll@ was enabled, adding a non-fatal error to
  --   the accumulator.
  --
  --   Control flow wise, non-fatal errors act like warnings: they are added
  --   to the accumulator and parsing continues. This allows GHC to report
  --   more than one parse error per file.
  --
  addError :: SrcSpan -> SDoc -> m ()
  -- | Add a warning to the accumulator.
  --   Use 'getMessages' to get the accumulated warnings.
  addWarning :: WarningFlag -> SrcSpan -> SDoc -> m ()
  -- | Add a fatal error. This will be the last error reported by the parser, and
  --   the parser will not produce any result, ending in a 'PFailed' state.
  addFatalError :: SrcSpan -> SDoc -> m a
  -- | Check if a given flag is currently set in the bitmap.
  getBit :: ExtBits -> m Bool
  -- | Given a location and a list of AddAnn, apply them all to the location.
  addAnnotation :: SrcSpan          -- SrcSpan of enclosing AST construct
                -> AnnKeywordId     -- The first two parameters are the key
                -> SrcSpan          -- The location of the keyword itself
                -> m ()

appendError
  :: SrcSpan
  -> SDoc
  -> (DynFlags -> Messages)
  -> (DynFlags -> Messages)
appendError srcspan msg m =
  \d ->
    let (ws, es) = m d
        errormsg = mkErrMsg d srcspan alwaysQualify msg
        es' = es `snocBag` errormsg
    in (ws, es')

appendWarning
  :: ParserFlags
  -> WarningFlag
  -> SrcSpan
  -> SDoc
  -> (DynFlags -> Messages)
  -> (DynFlags -> Messages)
appendWarning o option srcspan warning m =
  \d ->
    let (ws, es) = m d
        warning' = makeIntoWarning (Reason option) $
           mkWarnMsg d srcspan alwaysQualify warning
        ws' = if warnopt option o then ws `snocBag` warning' else ws
    in (ws', es)

instance MonadP P where
  addError srcspan msg
   = P $ \s@PState{messages=m} ->
             POk s{messages=appendError srcspan msg m} ()
  addWarning option srcspan warning
   = P $ \s@PState{messages=m, options=o} ->
             POk s{messages=appendWarning o option srcspan warning m} ()
  addFatalError span msg =
    addError span msg >> P PFailed
  getBit ext = P $ \s -> let b =  ext `xtest` pExtsBitmap (options s)
                         in b `seq` POk s b
  addAnnotation (RealSrcSpan l _) a (RealSrcSpan v _) = do
    addAnnotationOnly l a v
    allocateCommentsP l
  addAnnotation _ _ _ = return ()

addAnnsAt :: MonadP m => SrcSpan -> [AddAnn] -> m ()
addAnnsAt l = mapM_ (\(AddAnn a v) -> addAnnotation l a v)

addTabWarning :: RealSrcSpan -> P ()
addTabWarning srcspan
 = P $ \s@PState{tab_first=tf, tab_count=tc, options=o} ->
       let tf' = if isJust tf then tf else Just srcspan
           tc' = tc + 1
           s' = if warnopt Opt_WarnTabs o
                then s{tab_first = tf', tab_count = tc'}
                else s
       in POk s' ()

mkTabWarning :: PState -> DynFlags -> Maybe ErrMsg
mkTabWarning PState{tab_first=tf, tab_count=tc} d =
  let middle = if tc == 1
        then text ""
        else text ", and in" <+> speakNOf (tc - 1) (text "further location")
      message = text "Tab character found here"
                <> middle
                <> text "."
                $+$ text "Please use spaces instead."
  in fmap (\s -> makeIntoWarning (Reason Opt_WarnTabs) $
                 mkWarnMsg d (RealSrcSpan s Nothing) alwaysQualify message) tf

-- | Get a bag of the errors that have been accumulated so far.
--   Does not take -Werror into account.
getErrorMessages :: PState -> DynFlags -> ErrorMessages
getErrorMessages PState{messages=m} d =
  let (_, es) = m d in es

-- | Get the warnings and errors accumulated so far.
--   Does not take -Werror into account.
getMessages :: PState -> DynFlags -> Messages
getMessages p@PState{messages=m} d =
  let (ws, es) = m d
      tabwarning = mkTabWarning p d
      ws' = maybe ws (`consBag` ws) tabwarning
  in (ws', es)

getContext :: P [LayoutContext]
getContext = P $ \s@PState{context=ctx} -> POk s ctx

setContext :: [LayoutContext] -> P ()
setContext ctx = P $ \s -> POk s{context=ctx} ()

popContext :: P ()
popContext = P $ \ s@(PState{ buffer = buf, options = o, context = ctx,
                              last_len = len, last_loc = last_loc }) ->
  case ctx of
        (_:tl) ->
          POk s{ context = tl } ()
        []     ->
          unP (addFatalError (mkSrcSpanPs last_loc) (srcParseErr o buf len)) s

-- Push a new layout context at the indentation of the last token read.
pushCurrentContext :: GenSemic -> P ()
pushCurrentContext gen_semic = P $ \ s@PState{ last_loc=loc, context=ctx } ->
    POk s{context = Layout (srcSpanStartCol (psRealSpan loc)) gen_semic : ctx} ()

-- This is only used at the outer level of a module when the 'module' keyword is
-- missing.
pushModuleContext :: P ()
pushModuleContext = pushCurrentContext generateSemic

getOffside :: P (Ordering, Bool)
getOffside = P $ \s@PState{last_loc=loc, context=stk} ->
                let offs = srcSpanStartCol (psRealSpan loc) in
                let ord = case stk of
                            Layout n gen_semic : _ ->
                              --trace ("layout: " ++ show n ++ ", offs: " ++ show offs) $
                              (compare offs n, gen_semic)
                            _ ->
                              (GT, dontGenerateSemic)
                in POk s ord

-- ---------------------------------------------------------------------------
-- Construct a parse error

srcParseErr
  :: ParserFlags
  -> StringBuffer       -- current buffer (placed just after the last token)
  -> Int                -- length of the previous token
  -> MsgDoc
srcParseErr options buf len
  = if null token
         then text "parse error (possibly incorrect indentation or mismatched brackets)"
         else text "parse error on input" <+> quotes (text token)
              $$ ppWhen (not th_enabled && token == "$") -- #7396
                        (text "Perhaps you intended to use TemplateHaskell")
              $$ ppWhen (token == "<-")
                        (if mdoInLast100
                           then text "Perhaps you intended to use RecursiveDo"
                           else text "Perhaps this statement should be within a 'do' block?")
              $$ ppWhen (token == "=" && doInLast100) -- #15849
                        (text "Perhaps you need a 'let' in a 'do' block?"
                         $$ text "e.g. 'let x = 5' instead of 'x = 5'")
              $$ ppWhen (not ps_enabled && pattern == "pattern ") -- #12429
                        (text "Perhaps you intended to use PatternSynonyms")
  where token = lexemeToString (offsetBytes (-len) buf) len
        pattern = decodePrevNChars 8 buf
        last100 = decodePrevNChars 100 buf
        doInLast100 = "do" `isInfixOf` last100
        mdoInLast100 = "mdo" `isInfixOf` last100
        th_enabled = ThQuotesBit `xtest` pExtsBitmap options
        ps_enabled = PatternSynonymsBit `xtest` pExtsBitmap options

-- Report a parse failure, giving the span of the previous token as
-- the location of the error.  This is the entry point for errors
-- detected during parsing.
srcParseFail :: P a
srcParseFail = P $ \s@PState{ buffer = buf, options = o, last_len = len,
                            last_loc = last_loc } ->
    unP (addFatalError (mkSrcSpanPs last_loc) (srcParseErr o buf len)) s

-- A lexical error is reported at a particular position in the source file,
-- not over a token range.
lexError :: String -> P a
lexError str = do
  loc <- getRealSrcLoc
  (AI end buf) <- getInput
  reportLexError loc (psRealLoc end) buf str

-- -----------------------------------------------------------------------------
-- This is the top-level function: called from the parser each time a
-- new token is to be read from the input.

lexer, lexerDbg :: Bool -> (Located Token -> P a) -> P a

lexer queueComments cont = do
  alr <- getBit AlternativeLayoutRuleBit
  let lexTokenFun = if alr then lexTokenAlr else lexToken
  (L span tok) <- lexTokenFun
  --trace ("token: " ++ show tok) $ do

  if (queueComments && isDocComment tok)
    then queueComment (L (psRealSpan span) tok)
    else return ()

  if (queueComments && isComment tok)
    then queueComment (L (psRealSpan span) tok) >> lexer queueComments cont
    else cont (L (mkSrcSpanPs span) tok)

-- Use this instead of 'lexer' in GHC.Parser to dump the tokens for debugging.
lexerDbg queueComments cont = lexer queueComments contDbg
  where
    contDbg tok = trace ("token: " ++ show (unLoc tok)) (cont tok)

lexTokenAlr :: P (PsLocated Token)
lexTokenAlr = do mPending <- popPendingImplicitToken
                 t <- case mPending of
                      Nothing ->
                          do mNext <- popNextToken
                             t <- case mNext of
                                  Nothing -> lexToken
                                  Just next -> return next
                             alternativeLayoutRuleToken t
                      Just t ->
                          return t
                 setAlrLastLoc (getLoc t)
                 case unLoc t of
                     ITwhere -> setAlrExpectingOCurly (Just ALRLayoutWhere)
                     ITlet   -> setAlrExpectingOCurly (Just ALRLayoutLet)
                     ITof    -> setAlrExpectingOCurly (Just ALRLayoutOf)
                     ITlcase -> setAlrExpectingOCurly (Just ALRLayoutOf)
                     ITdo    -> setAlrExpectingOCurly (Just ALRLayoutDo)
                     ITmdo   -> setAlrExpectingOCurly (Just ALRLayoutDo)
                     ITrec   -> setAlrExpectingOCurly (Just ALRLayoutDo)
                     _       -> return ()
                 return t

alternativeLayoutRuleToken :: PsLocated Token -> P (PsLocated Token)
alternativeLayoutRuleToken t
    = do context <- getALRContext
         lastLoc <- getAlrLastLoc
         mExpectingOCurly <- getAlrExpectingOCurly
         transitional <- getBit ALRTransitionalBit
         justClosedExplicitLetBlock <- getJustClosedExplicitLetBlock
         setJustClosedExplicitLetBlock False
         let thisLoc = getLoc t
             thisCol = srcSpanStartCol (psRealSpan thisLoc)
             newLine = srcSpanStartLine (psRealSpan thisLoc) > srcSpanEndLine (psRealSpan lastLoc)
         case (unLoc t, context, mExpectingOCurly) of
             -- This case handles a GHC extension to the original H98
             -- layout rule...
             (ITocurly, _, Just alrLayout) ->
                 do setAlrExpectingOCurly Nothing
                    let isLet = case alrLayout of
                                ALRLayoutLet -> True
                                _ -> False
                    setALRContext (ALRNoLayout (containsCommas ITocurly) isLet : context)
                    return t
             -- ...and makes this case unnecessary
             {-
             -- I think our implicit open-curly handling is slightly
             -- different to John's, in how it interacts with newlines
             -- and "in"
             (ITocurly, _, Just _) ->
                 do setAlrExpectingOCurly Nothing
                    setNextToken t
                    lexTokenAlr
             -}
             (_, ALRLayout _ col : _ls, Just expectingOCurly)
              | (thisCol > col) ||
                (thisCol == col &&
                 isNonDecreasingIndentation expectingOCurly) ->
                 do setAlrExpectingOCurly Nothing
                    setALRContext (ALRLayout expectingOCurly thisCol : context)
                    setNextToken t
                    return (L thisLoc ITvocurly)
              | otherwise ->
                 do setAlrExpectingOCurly Nothing
                    setPendingImplicitTokens [L lastLoc ITvccurly]
                    setNextToken t
                    return (L lastLoc ITvocurly)
             (_, _, Just expectingOCurly) ->
                 do setAlrExpectingOCurly Nothing
                    setALRContext (ALRLayout expectingOCurly thisCol : context)
                    setNextToken t
                    return (L thisLoc ITvocurly)
             -- We do the [] cases earlier than in the spec, as we
             -- have an actual EOF token
             (ITeof, ALRLayout _ _ : ls, _) ->
                 do setALRContext ls
                    setNextToken t
                    return (L thisLoc ITvccurly)
             (ITeof, _, _) ->
                 return t
             -- the other ITeof case omitted; general case below covers it
             (ITin, _, _)
              | justClosedExplicitLetBlock ->
                 return t
             (ITin, ALRLayout ALRLayoutLet _ : ls, _)
              | newLine ->
                 do setPendingImplicitTokens [t]
                    setALRContext ls
                    return (L thisLoc ITvccurly)
             -- This next case is to handle a transitional issue:
             (ITwhere, ALRLayout _ col : ls, _)
              | newLine && thisCol == col && transitional ->
                 do addWarning Opt_WarnAlternativeLayoutRuleTransitional
                               (mkSrcSpanPs thisLoc)
                               (transitionalAlternativeLayoutWarning
                                    "`where' clause at the same depth as implicit layout block")
                    setALRContext ls
                    setNextToken t
                    -- Note that we use lastLoc, as we may need to close
                    -- more layouts, or give a semicolon
                    return (L lastLoc ITvccurly)
             -- This next case is to handle a transitional issue:
             (ITvbar, ALRLayout _ col : ls, _)
              | newLine && thisCol == col && transitional ->
                 do addWarning Opt_WarnAlternativeLayoutRuleTransitional
                               (mkSrcSpanPs thisLoc)
                               (transitionalAlternativeLayoutWarning
                                    "`|' at the same depth as implicit layout block")
                    setALRContext ls
                    setNextToken t
                    -- Note that we use lastLoc, as we may need to close
                    -- more layouts, or give a semicolon
                    return (L lastLoc ITvccurly)
             (_, ALRLayout _ col : ls, _)
              | newLine && thisCol == col ->
                 do setNextToken t
                    let loc = psSpanStart thisLoc
                        zeroWidthLoc = mkPsSpan loc loc
                    return (L zeroWidthLoc ITsemi)
              | newLine && thisCol < col ->
                 do setALRContext ls
                    setNextToken t
                    -- Note that we use lastLoc, as we may need to close
                    -- more layouts, or give a semicolon
                    return (L lastLoc ITvccurly)
             -- We need to handle close before open, as 'then' is both
             -- an open and a close
             (u, _, _)
              | isALRclose u ->
                 case context of
                 ALRLayout _ _ : ls ->
                     do setALRContext ls
                        setNextToken t
                        return (L thisLoc ITvccurly)
                 ALRNoLayout _ isLet : ls ->
                     do let ls' = if isALRopen u
                                     then ALRNoLayout (containsCommas u) False : ls
                                     else ls
                        setALRContext ls'
                        when isLet $ setJustClosedExplicitLetBlock True
                        return t
                 [] ->
                     do let ls = if isALRopen u
                                    then [ALRNoLayout (containsCommas u) False]
                                    else []
                        setALRContext ls
                        -- XXX This is an error in John's code, but
                        -- it looks reachable to me at first glance
                        return t
             (u, _, _)
              | isALRopen u ->
                 do setALRContext (ALRNoLayout (containsCommas u) False : context)
                    return t
             (ITin, ALRLayout ALRLayoutLet _ : ls, _) ->
                 do setALRContext ls
                    setPendingImplicitTokens [t]
                    return (L thisLoc ITvccurly)
             (ITin, ALRLayout _ _ : ls, _) ->
                 do setALRContext ls
                    setNextToken t
                    return (L thisLoc ITvccurly)
             -- the other ITin case omitted; general case below covers it
             (ITcomma, ALRLayout _ _ : ls, _)
              | topNoLayoutContainsCommas ls ->
                 do setALRContext ls
                    setNextToken t
                    return (L thisLoc ITvccurly)
             (ITwhere, ALRLayout ALRLayoutDo _ : ls, _) ->
                 do setALRContext ls
                    setPendingImplicitTokens [t]
                    return (L thisLoc ITvccurly)
             -- the other ITwhere case omitted; general case below covers it
             (_, _, _) -> return t

transitionalAlternativeLayoutWarning :: String -> SDoc
transitionalAlternativeLayoutWarning msg
    = text "transitional layout will not be accepted in the future:"
   $$ text msg

isALRopen :: Token -> Bool
isALRopen ITcase          = True
isALRopen ITif            = True
isALRopen ITthen          = True
isALRopen IToparen        = True
isALRopen ITobrack        = True
isALRopen ITocurly        = True
-- GHC Extensions:
isALRopen IToubxparen     = True
isALRopen _               = False

isALRclose :: Token -> Bool
isALRclose ITof     = True
isALRclose ITthen   = True
isALRclose ITelse   = True
isALRclose ITcparen = True
isALRclose ITcbrack = True
isALRclose ITccurly = True
-- GHC Extensions:
isALRclose ITcubxparen = True
isALRclose _        = False

isNonDecreasingIndentation :: ALRLayout -> Bool
isNonDecreasingIndentation ALRLayoutDo = True
isNonDecreasingIndentation _           = False

containsCommas :: Token -> Bool
containsCommas IToparen = True
containsCommas ITobrack = True
-- John doesn't have {} as containing commas, but records contain them,
-- which caused a problem parsing Cabal's Distribution.Simple.InstallDirs
-- (defaultInstallDirs).
containsCommas ITocurly = True
-- GHC Extensions:
containsCommas IToubxparen = True
containsCommas _        = False

topNoLayoutContainsCommas :: [ALRContext] -> Bool
topNoLayoutContainsCommas [] = False
topNoLayoutContainsCommas (ALRLayout _ _ : ls) = topNoLayoutContainsCommas ls
topNoLayoutContainsCommas (ALRNoLayout b _ : _) = b

lexToken :: P (PsLocated Token)
lexToken = do
  inp@(AI loc1 buf) <- getInput
  sc <- getLexState
  exts <- getExts
  case alexScanUser exts inp sc of
    AlexEOF -> do
        let span = mkPsSpan loc1 loc1
        setEofPos (psRealSpan span)
        setLastToken span 0
        return (L span ITeof)
    AlexError (AI loc2 buf) ->
        reportLexError (psRealLoc loc1) (psRealLoc loc2) buf "lexical error"
    AlexSkip inp2 _ -> do
        setInput inp2
        lexToken
    AlexToken inp2@(AI end buf2) _ t -> do
        setInput inp2
        let span = mkPsSpan loc1 end
        let bytes = byteDiff buf buf2
        span `seq` setLastToken span bytes
        lt <- t span buf bytes
        let lt' = unLoc lt
        unless (isComment lt') (setLastTk lt')
        return lt

reportLexError :: RealSrcLoc -> RealSrcLoc -> StringBuffer -> [Char] -> P a
reportLexError loc1 loc2 buf str
  | atEnd buf = failLocMsgP loc1 loc2 (str ++ " at end of input")
  | otherwise =
  let c = fst (nextChar buf)
  in if c == '\0' -- decoding errors are mapped to '\0', see utf8DecodeChar#
     then failLocMsgP loc2 loc2 (str ++ " (UTF-8 decoding error)")
     else failLocMsgP loc1 loc2 (str ++ " at character " ++ show c)

lexTokenStream :: StringBuffer -> RealSrcLoc -> DynFlags -> ParseResult [Located Token]
lexTokenStream buf loc dflags = unP go initState{ options = opts' }
    where dflags' = gopt_set (gopt_unset dflags Opt_Haddock) Opt_KeepRawTokenStream
          initState@PState{ options = opts } = mkPState dflags' buf loc
          opts' = opts{ pExtsBitmap = complement (xbit UsePosPragsBit) .&. pExtsBitmap opts }
          go = do
            ltok <- lexer False return
            case ltok of
              L _ ITeof -> return []
              _ -> liftM (ltok:) go

linePrags = Map.singleton "line" linePrag

fileHeaderPrags = Map.fromList([("options", lex_string_prag IToptions_prag),
                                 ("options_ghc", lex_string_prag IToptions_prag),
                                 ("options_haddock", lex_string_prag ITdocOptions),
                                 ("language", token ITlanguage_prag),
                                 ("include", lex_string_prag ITinclude_prag)])

ignoredPrags = Map.fromList (map ignored pragmas)
               where ignored opt = (opt, nested_comment lexToken)
                     impls = ["hugs", "nhc98", "jhc", "yhc", "catch", "derive"]
                     options_pragmas = map ("options_" ++) impls
                     -- CFILES is a hugs-only thing.
                     pragmas = options_pragmas ++ ["cfiles", "contract"]

oneWordPrags = Map.fromList [
     ("rules", rulePrag),
     ("inline",
         strtoken (\s -> (ITinline_prag (SourceText s) Inline FunLike))),
     ("inlinable",
         strtoken (\s -> (ITinline_prag (SourceText s) Inlinable FunLike))),
     ("inlineable",
         strtoken (\s -> (ITinline_prag (SourceText s) Inlinable FunLike))),
                                    -- Spelling variant
     ("notinline",
         strtoken (\s -> (ITinline_prag (SourceText s) NoInline FunLike))),
     ("specialize", strtoken (\s -> ITspec_prag (SourceText s))),
     ("source", strtoken (\s -> ITsource_prag (SourceText s))),
     ("warning", strtoken (\s -> ITwarning_prag (SourceText s))),
     ("deprecated", strtoken (\s -> ITdeprecated_prag (SourceText s))),
     ("scc", strtoken (\s -> ITscc_prag (SourceText s))),
     ("generated", strtoken (\s -> ITgenerated_prag (SourceText s))),
     ("core", strtoken (\s -> ITcore_prag (SourceText s))),
     ("unpack", strtoken (\s -> ITunpack_prag (SourceText s))),
     ("nounpack", strtoken (\s -> ITnounpack_prag (SourceText s))),
     ("ann", strtoken (\s -> ITann_prag (SourceText s))),
     ("minimal", strtoken (\s -> ITminimal_prag (SourceText s))),
     ("overlaps", strtoken (\s -> IToverlaps_prag (SourceText s))),
     ("overlappable", strtoken (\s -> IToverlappable_prag (SourceText s))),
     ("overlapping", strtoken (\s -> IToverlapping_prag (SourceText s))),
     ("incoherent", strtoken (\s -> ITincoherent_prag (SourceText s))),
     ("ctype", strtoken (\s -> ITctype (SourceText s))),
     ("complete", strtoken (\s -> ITcomplete_prag (SourceText s))),
     ("column", columnPrag)
     ]

twoWordPrags = Map.fromList [
     ("inline conlike",
         strtoken (\s -> (ITinline_prag (SourceText s) Inline ConLike))),
     ("notinline conlike",
         strtoken (\s -> (ITinline_prag (SourceText s) NoInline ConLike))),
     ("specialize inline",
         strtoken (\s -> (ITspec_inline_prag (SourceText s) True))),
     ("specialize notinline",
         strtoken (\s -> (ITspec_inline_prag (SourceText s) False)))
     ]

dispatch_pragmas :: Map String Action -> Action
dispatch_pragmas prags span buf len = case Map.lookup (clean_pragma (lexemeToString buf len)) prags of
                                       Just found -> found span buf len
                                       Nothing -> lexError "unknown pragma"

known_pragma :: Map String Action -> AlexAccPred ExtsBitmap
known_pragma prags _ (AI _ startbuf) _ (AI _ curbuf)
 = isKnown && nextCharIsNot curbuf pragmaNameChar
    where l = lexemeToString startbuf (byteDiff startbuf curbuf)
          isKnown = isJust $ Map.lookup (clean_pragma l) prags
          pragmaNameChar c = isAlphaNum c || c == '_'

clean_pragma :: String -> String
clean_pragma prag = canon_ws (map toLower (unprefix prag))
                    where unprefix prag' = case stripPrefix "{-#" prag' of
                                             Just rest -> rest
                                             Nothing -> prag'
                          canonical prag' = case prag' of
                                              "noinline" -> "notinline"
                                              "specialise" -> "specialize"
                                              "constructorlike" -> "conlike"
                                              _ -> prag'
                          canon_ws s = unwords (map canonical (words s))



{-
%************************************************************************
%*                                                                      *
        Helper functions for generating annotations in the parser
%*                                                                      *
%************************************************************************
-}

-- | Encapsulated call to addAnnotation, requiring only the SrcSpan of
--   the AST construct the annotation belongs to; together with the
--   AnnKeywordId, this is the key of the annotation map.
--
--   This type is useful for places in the parser where it is not yet
--   known what SrcSpan an annotation should be added to.  The most
--   common situation is when we are parsing a list: the annotations
--   need to be associated with the AST element that *contains* the
--   list, not the list itself.  'AddAnn' lets us defer adding the
--   annotations until we finish parsing the list and are now parsing
--   the enclosing element; we then apply the 'AddAnn' to associate
--   the annotations.  Another common situation is where a common fragment of
--   the AST has been factored out but there is no separate AST node for
--   this fragment (this occurs in class and data declarations). In this
--   case, the annotation belongs to the parent data declaration.
--
--   The usual way an 'AddAnn' is created is using the 'mj' ("make jump")
--   function, and then it can be discharged using the 'ams' function.
data AddAnn = AddAnn AnnKeywordId SrcSpan

addAnnotationOnly :: RealSrcSpan -> AnnKeywordId -> RealSrcSpan -> P ()
addAnnotationOnly l a v = P $ \s -> POk s {
  annotations = ((l,a), [v]) : annotations s
  } ()

-- |Given a 'SrcSpan' that surrounds a 'HsPar' or 'HsParTy', generate
-- 'AddAnn' values for the opening and closing bordering on the start
-- and end of the span
mkParensApiAnn :: SrcSpan -> [AddAnn]
mkParensApiAnn (UnhelpfulSpan _)  = []
mkParensApiAnn (RealSrcSpan ss _) = [AddAnn AnnOpenP lo,AddAnn AnnCloseP lc]
  where
    f = srcSpanFile ss
    sl = srcSpanStartLine ss
    sc = srcSpanStartCol ss
    el = srcSpanEndLine ss
    ec = srcSpanEndCol ss
    lo = RealSrcSpan (mkRealSrcSpan (realSrcSpanStart ss)        (mkRealSrcLoc f sl (sc+1))) Nothing
    lc = RealSrcSpan (mkRealSrcSpan (mkRealSrcLoc f el (ec - 1)) (realSrcSpanEnd ss))        Nothing

queueComment :: RealLocated Token -> P()
queueComment c = P $ \s -> POk s {
  comment_q = commentToAnnotation c : comment_q s
  } ()

-- | Go through the @comment_q@ in @PState@ and remove all comments
-- that belong within the given span
allocateCommentsP :: RealSrcSpan -> P ()
allocateCommentsP ss = P $ \s ->
  let (comment_q', newAnns) = allocateComments ss (comment_q s) in
    POk s {
       comment_q = comment_q'
     , annotations_comments = newAnns ++ (annotations_comments s)
     } ()

allocateComments
  :: RealSrcSpan
  -> [RealLocated AnnotationComment]
  -> ([RealLocated AnnotationComment], [(RealSrcSpan,[RealLocated AnnotationComment])])
allocateComments ss comment_q =
  let
    (before,rest)  = break (\(L l _) -> isRealSubspanOf l ss) comment_q
    (middle,after) = break (\(L l _) -> not (isRealSubspanOf l ss)) rest
    comment_q' = before ++ after
    newAnns = if null middle then []
                             else [(ss,middle)]
  in
    (comment_q', newAnns)


commentToAnnotation :: RealLocated Token -> RealLocated AnnotationComment
commentToAnnotation (L l (ITdocCommentNext s))  = L l (AnnDocCommentNext s)
commentToAnnotation (L l (ITdocCommentPrev s))  = L l (AnnDocCommentPrev s)
commentToAnnotation (L l (ITdocCommentNamed s)) = L l (AnnDocCommentNamed s)
commentToAnnotation (L l (ITdocSection n s))    = L l (AnnDocSection n s)
commentToAnnotation (L l (ITdocOptions s))      = L l (AnnDocOptions s)
commentToAnnotation (L l (ITlineComment s))     = L l (AnnLineComment s)
commentToAnnotation (L l (ITblockComment s))    = L l (AnnBlockComment s)
commentToAnnotation _                           = panic "commentToAnnotation"

-- ---------------------------------------------------------------------

isComment :: Token -> Bool
isComment (ITlineComment     _)   = True
isComment (ITblockComment    _)   = True
isComment _ = False

isDocComment :: Token -> Bool
isDocComment (ITdocCommentNext  _)   = True
isDocComment (ITdocCommentPrev  _)   = True
isDocComment (ITdocCommentNamed _)   = True
isDocComment (ITdocSection      _ _) = True
isDocComment (ITdocOptions      _)   = True
isDocComment _ = False
}
