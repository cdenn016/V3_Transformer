# Multilingual Figure Correctness Design

## Status and decision

This design repairs the figure path reported during training and makes Japanese and Arabic run artifacts display honest native-script text. The selected approach fixes the underlying data and rendering contracts rather than suppressing warnings. Existing English figures retain their current linguistic diagnostics. Japanese and Arabic figures use the same quantitative model and geometry diagnostics, but English-only BPE case and function/content taxonomies are marked unavailable and are not emitted as if they were language-general measurements.

The user approved native Japanese labels, correctly shaped right-to-left Arabic labels, and removal of misleading English-only linguistic diagnostics for `wiki-ja` and `wiki-ar`.

## Observed defects

Several trajectory functions set their x-axis directly to the minimum and maximum recorded step. A history containing exactly one observation therefore requests identical lower and upper bounds. Matplotlib expands the interval automatically and emits the reported singular-transformation warning. The same pattern occurs outside the four reported lines, so repairing only those call sites would leave equivalent warnings in other one-point figures.

The gradient-decomposition labels render a calligraphic loss symbol through `\mathcal{L}`. Under a Matplotlib font configuration that falls back through STIX, each legend and axis-label render can report `Substituting symbol L from STIXNonUnicode`. The calligraphic styling is not needed to distinguish this loss from another plotted quantity.

The tokenized cache on this machine records both `wiki-ja` and `wiki-ar` as `tiktoken_cl100k`. The live dataset registry includes `wiki-ja` and `wiki-en` in its cl100k set but omits `wiki-ar`. As a result, Arabic cache lookup constructs a nonexistent GPT-2-tagged filename, vocabulary validation assumes the wrong bound, and the figure decoder selects GPT-2 instead of the tokenizer that produced the cache. This is a provenance and correctness defect, not a display preference.

Matplotlib's default DejaVu Sans configuration lacks Japanese glyphs and renders boxes. The installed Arabic fonts contain the glyphs, but Matplotlib's standard text renderer does not perform the Arabic joining and bidirectional layout required for readable right-to-left text. Font fallback alone therefore repairs Japanese but leaves Arabic letters disconnected and ordered incorrectly.

The current UMAP and category-confusion diagnostics also assume English. Their categories distinguish lowercase and capitalized word starts, and their function-word set contains English stopwords only. Applying those labels to Japanese or Arabic turns nearly every alphabetic token into an English `content` token and gives Japanese fragments an English word-boundary marker. Those values do not measure the advertised linguistic properties.

## Approaches considered

A cosmetic approach would catch or filter the warnings, leave tokenizer selection unchanged, and replace missing native glyphs with token IDs. This is small but wrong: the Arabic decoder would still disagree with the cache, and warning suppression would hide zero-width axes rather than define them.

A transliteration approach would convert Japanese and Arabic labels to Latin text and retain the English taxonomies. It avoids font and shaping dependencies but discards the text actually represented by the token and still does not make the English linguistic categories valid.

The selected approach repairs each contract at its source. It centralizes finite x-axis bounds, uses a font-safe loss label, binds `wiki-ar` to its cl100k cache and decoder, adds Unicode-capable font fallback and Arabic shaping at the final display boundary, and disables only the English-only diagnostics for Japanese and Arabic. Decoded logical text remains unchanged for classification, sorting, provenance, and data processing; right-to-left transformation occurs only when a string is handed to Matplotlib.

## Axis and math-label behavior

A shared axis helper will accept the recorded x values, discard nonfinite entries, and set explicit finite limits. Distinct endpoints are used unchanged. A single finite endpoint receives deterministic symmetric padding based on its magnitude, with a minimum one-step pad. Empty or wholly nonfinite inputs leave Matplotlib autoscaling untouched. Every data-derived `set_xlim(min, max)` trajectory call in `vfe3/viz/figures.py` will use this helper, including the reported history, model-channel, gradient-decomposition, and dashboard paths as well as equivalent kappa and co-descent paths.

Gradient-decomposition labels will use a plain mathematical `L` for the optimized loss. The notation remains `||nabla L||_2`, but it no longer requests a calligraphic glyph from a fallback math font. Fixed geometric axis intervals and unrelated math notation remain unchanged.

## Tokenizer provenance

`wiki-ar` will join `wiki-ja` and `wiki-en` in the cl100k dataset registry. Cache paths, tokenizer tags, vocabulary bounds, text decoding, bits-per-character decoding, artifact provenance, and any code that consumes the central registry will consequently agree with the existing Arabic cache metadata. No cache will be rewritten, and no checkpoint format or model objective will change.

Tests will pin the tokenizer tag, cache filename, vocabulary size, and decoder encoding choice for both `wiki-ja` and `wiki-ar`. The tests will mock tokenizer construction where possible so correctness does not depend on downloading tokenizer tables during the suite.

## Native-script display boundary

Publication style will define an ordered sans-serif fallback list containing common CJK fonts, `Yu Gothic` for the current Windows environment, Noto Arabic families, and DejaVu Sans for Latin text. Missing optional system fonts must not be treated as an error when a later fallback covers the text. Japanese labels remain ordinary Unicode strings and rely on glyph fallback; no romanization is introduced.

Arabic display strings will be reshaped into contextual presentation forms and passed through the Unicode bidirectional display algorithm immediately before Matplotlib receives them. The project will declare the small `arabic-reshaper` and `python-bidi` packages as visualization dependencies required by the default-on multilingual figure path. The display helper will activate only when Arabic code points are present, so English and Japanese strings are byte-for-byte unchanged. Mixed Arabic, punctuation, numerals, and Latin prefixes will be transformed as one complete label so their visual order remains coherent.

Decoded single-token fragments that contain the Unicode replacement character or nonprintable data cannot be presented as valid native text. Vocabulary figures will show a stable token-ID fallback for those fragments instead of rendering replacement glyphs. Cluster-label enrichment will continue to drop invalid fragments. Length limits will be applied to logical Unicode text before Arabic display shaping so truncation remains deterministic.

## Language-valid diagnostics

The report driver already knows the dataset name. It will derive whether English linguistic diagnostics are valid and pass that decision to token-aware figures. `wikitext-2`, `wikitext-103`, and `wiki-en` retain BPE structure, function/content silhouettes, belief-category separation, and category-confusion figures. `wiki-ja` and `wiki-ar` omit belief-category separation and vocabulary category confusion, and controlled UMAP sidecars store the associated taxonomy metrics as unavailable with an explicit non-English reason.

UMAP cluster discovery, cluster size, noise fraction, projection stability, relative-position diagnostics, sequence-identity diagnostics, and native geometry measurements remain available for every language. Japanese and Arabic cluster legends use decoded native text without the English continuation-subword middle dot. The report footer will state when English linguistic taxonomies were disabled so an absent category figure cannot be mistaken for a failed plot.

Direct figure callers retain the current English behavior by default for compatibility. The report path supplies the dataset-aware switch explicitly. Cross-run vocabulary comparison will apply the same policy and will not generate an English category-confusion matrix for Japanese or Arabic arms.

## Dependencies and failure behavior

The shaping packages belong with visualization dependencies because native Arabic is otherwise rendered incorrectly. The code will import them lazily inside the Arabic display helper so importing the core model remains lightweight. If an environment bypasses declared dependencies and attempts to render Arabic without them, the figure must fail with a concise message naming the missing visualization requirement rather than silently writing disconnected text. The report's existing best-effort boundary may skip that affected figure while continuing other diagnostics.

Font availability is checked by actual render tests on the configured environment. The implementation will not embed or redistribute proprietary fonts. On this Windows system, `Yu Gothic` and Noto Arabic fonts are already installed.

## Tests and acceptance criteria

Tests will first reproduce the four reported one-point x-axis warnings and the equivalent unreported min/max paths. After the repair, warning-as-error rendering must succeed and each one-point axis must have finite, strictly ordered limits containing the recorded step. Multi-point histories must retain their exact endpoint limits.

Tokenizer tests must show that Arabic and Japanese resolve to cl100k cache tags, 100,277-token vocabulary bounds, and cl100k decoders. Figure tests must show that Japanese survives unchanged, Arabic is reshaped and reordered for display, invalid token fragments fall back to IDs, and the publication font list includes working CJK and Arabic families. A rendered Japanese sample must produce no missing-glyph warning in the current environment, while an Arabic sample must use connected presentation forms rather than isolated logical code points.

Report tests must show that English runs still emit their existing category diagnostics, Japanese and Arabic runs skip them deliberately, and UMAP sidecars retain explicit unavailable reasons without losing the language-general metrics. Existing visualization, reporting, data, controlled-UMAP, and full-repository tests must remain green. The required dated edit record will state the exact machine-readable verification counts.

No training configuration values, cached token streams, model weights, objectives, or historical run artifacts will be modified. Existing figures remain historical and can be regenerated from their saved run directories after the corrected code and dependencies land.
