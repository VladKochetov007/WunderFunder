# BNP Paribas Kaggle talk (Semenov transcript): transferable techniques

Source: `writeups/semenov.txt` (1,093 lines, read in full). Line numbers below refer to that file.

## Scope check and caveats

- **The transcript does not describe the problem in the mining brief.** It describes the BNP Paribas Kaggle competition: about 228,000 rows, about 130 anonymized features named V1 to V130 (L86-89, L107), and a binary 0/1 target (L90-91). It is tabular, with no time sequences, no 112-column layout, no multi-rate clocks, no piecewise-constant labels, no causal GRU and no weighted Pearson metric. Nothing below claims those items appear in the source.
- **The transcript is machine Russian ASR with many errors** (for example "Демина" for "деноминатора", "of Evidence" for "Weight of Evidence", and garbled library and team names). Quotes are approximate. The library name and the first-place team's method are not recoverable from the text.
- The speaker says the first-place team decoded the features (L111-114), but no decoding method is given. Treat this as unverified hearsay.
- The repo git log already has a commit titled "BNP lesson: linear dependency EDA" (1fe85e21). Check it before repeating technique A.

---

## 1. Exact linear dependencies between columns

- **What:** Some columns are exact linear combinations of others, for example one column equals the sum of two or five others. Regressing each column on the rest and checking whether the residual goes to zero finds them automatically.
- **Cheap test (CPU, under 1 h):** Per column, run least squares on all other columns and record R^2 and residual norm. Also take the SVD or QR rank of the full 112-column matrix on a sample. Flag columns with residual near zero and report the rank gap.
- **Evidence:**
  - L126-127: "есть признак который ... равен ... сумме двух других" (a feature equal to the sum of two others).
  - L137-145: "построил линейную регрессию предсказываю какой-то другой признак и смотрел на ошибку, если ошибка стремится к нулю, значит есть зависимость ... можно было автоматически находить все такие линейные зависимости" (built a linear regression to predict one feature from the others and checked the error; if it goes to zero there is a dependence; this finds all such dependencies automatically).
  - L262-264: "половину признаков можно было просто сразу же выкинуть" (half of the features could be dropped immediately).

## 2. Discrete values plus noise, then rescaling; recovering the scale ("denominator")

- **What:** Each feature was an integer grid value with added random noise, then rescaled into a fixed range. Its histogram looks smeared and continuous. The step size, found as the minimum gap between neighbouring values, undoes the rescaling and recovers the integer grid.
- **Cheap test (CPU, under 1 h):** For each column, take sorted unique values and their gaps. Find the smallest recurrent step and test whether values sit on a lattice k*step + offset. Compare that with a histogram. Note that a continuous column such as age will not pass this test. Trees are invariant to monotone rescaling, so the grid matters only for finding structure and for differences.
- **Evidence:**
  - L182-194: "дискретное значение плюс шум ... каждому добавили случайный ... шум ... потом взяли и это нормировать от нуля до два(дцати)" (a discrete value plus noise; random noise added to each value, then normalized to a range).
  - L201-203: "поиск деминатора ... коэффициентов, на которые потом всё это домножить" (searching for the denominator, meaning the coefficients by which everything is then multiplied).
  - L861-862: "минимальная разность между соседними значениями" (the minimum difference between neighbouring values).
  - L218: "дереву плевать что мы сделали" (the tree does not care what we did to the feature).
  - L212-215: "у них могут не быть постоянные разности между соседними числами" (numeric features such as age may not have constant gaps). This is a caveat: the lattice test should not be applied blindly.

## 3. Recovering hidden row order from column differences

- **What:** Two time-like columns were subtracted, and sorting rows by the difference recovered the order of the records. The winning approach then added the row's sequence index as a feature.
- **Cheap test (CPU, under 1 h):** For each pair of candidate time-like columns, compute the difference and check whether it is monotone or has a block structure in the data's implicit order. Use this only as a data-structure diagnostic. Any recovered clock must be used causally in our pipeline.
- **Evidence:**
  - L790-799: "второе время ... разность ... отсортировать ... сначала идут только нолики, а потом единички ... добавили порядковый номер" (the second time column minus the first; sorting by the difference gives zeros first, then ones; they added the row's order number).
  - L249-256: a column with about 360 unique values behaves like a day or a time scale. "это некие дни, некий тайм ... временную шкалу" (these are days, some kind of time scale).

## 4. High-cardinality grouping columns: subsetting by a level

- **What:** Selecting rows with one fixed value of a categorical column cut the number of distinct levels in other columns by about an order of magnitude. That column is a strong grouping key.
- **Cheap test (CPU, under 1 h):** For every candidate key, count unique values in the subset and the total, and compute the eta^2 of the target between groups. Rank keys by distinct-count reduction and by eta^2.
- **Evidence:**
  - L293-304: "выбрать только те ... объекты у которых ... v22 ... равнялся ... значению 0" (selecting only the objects where v22 equals a given value, 0).
  - L309-317: "число различных уровней сильно сокращается, становится на порядок меньше ... признак v22 ... очень хорошо группирует всё" (the number of distinct levels drops by an order of magnitude; v22 groups everything very well).

## 5. Target (mean) encoding with count-based shrinkage

- **What:** Replace a categorical level with the target mean in that level, shrunk toward the global mean in proportion to the level's sample count. Tune the shrinkage constant alpha by cross-validation.
- **Cheap test (CPU, under 1 h):** Compute enc = (n_g * mean_g + alpha * global_mean) / (n_g + alpha) using only the training history, with alpha in {1, 5, 10, 50}. Check the gap between training and holdout, and use the same nested scheme as technique 7.
- **Evidence:**
  - L340-354 and L356-365: "если встретится какой-то новый объект ... этот признак равен единице, то таргет скорее всего будет равен единичке ... создам новую переменную ... среднее значение таргета" (if a new object has this feature equal to 1, its target is likely 1; create a new variable holding the mean target for each level).
  - L434-449: "смотрим сколько у нас есть таких записей ... Альфа ... например ... 10 ... если очень большое ... среднее ... стремится к среднему значению таргета" (we count how many records have the value; alpha is a coefficient, for example 10; a large count pulls the mean toward the group's own mean).
  - L450-457: "если значений очень мало, мы ничего не знаем, говорим что скорее всего это будет среднее ... глобальное" (with very few values we fall back on the global mean).
  - L461-463: "Единственное что нужно подбирать коэффициент альфа, его можно подбирать ... простой кросс-валидацией" (the only parameter to tune is alpha, by simple cross-validation).
  - L416-429: a level seen only once with target 0 gives a mean of exactly 0, and "на тесте это может оказаться всё что угодно" (on test it can turn out to be anything). This is the overfitting case the shrinkage addresses.

## 6. Weight of evidence (log-odds) encoding, regularized

- **What:** For a binary target, encode each level as the log of the ratio of positives to negatives, regularized so that small groups do not blow up.
- **Cheap test (CPU, under 1 h):** Compute WoE per level with additive smoothing (for example, add 0.5 to the counts). Compare its holdout AUC or log-loss against technique 5 on the same folds.
- **Evidence:**
  - L700-703: "есть такая вещь ... of Evidence ... некий логарифм от ... единичек в группе на ... число всего" (there is a thing called Weight of Evidence; a log of the share of ones in the group).
  - L709-711: "здесь такая регуляризация, брался логарифм, что здесь очень хорошо сработало" (the regularization here takes a log, which worked very well here).

## 7. Out-of-fold and nested encoding to prevent second-level leakage (the central validity lesson)

- **What:** A target-derived feature for a row must be built only from data that excludes that row's own fold and the held-out fold. The speaker's key point is that a naive scheme that encodes the full training set and then cross-validates the second level leaks silently, and the leaderboard does not reveal it. The fix is nested: inside each outer training fold, compute encodings with inner folds, and encode the outer held-out fold using only the outer training fold.
- **Cheap test (CPU, under 1 h):** Build the nested encoder for one target-encoded feature. Compare outer-CV score under the naive scheme and under the nested scheme. If the naive score is clearly higher while holdout is not, the gap is leakage. Our causal sequence version should be built so that encodings use only history before the prediction time.
- **Evidence:**
  - L481-487: "для третьего фолда берём и считаем средние значения, используя первый и второй" (for fold 3 we compute means using folds 1 and 2).
  - L511-533: "когда мы обучаем первый и второй фолд ... эти признаки ... используют третий фолд ... мы как бы это не должны видеть, а мы здесь неявно это видим" (when training on folds 1 and 2, these features use fold 3, which we should not see; we see it implicitly). The speaker calls this "очень важный момент" (a very important point).
  - L536-544: "если мы так сделаем ... на лидерборд это никак не будет отражаться" (if we do it this way, it will not show on the leaderboard in any way).
  - L544-550: "мы должны считать ... средние ... ещё внутри каждого фолда отдельно" (we must compute the means separately inside each fold).
  - L553-563: "берём первые два фолда, его ещё раз разбиваем на ... три ... внутри этого фолда считаем средние" (take the first two folds, split them again into three, and compute means inside).
  - L583-589: "только если так делать ... только тогда можно получить абсолютно верный результат ... нигде даже не явно ... не использовали информацию ... тестовой выборки" (only if you do it this way can you get a fully correct result, with no implicit use of test information anywhere).

## 8. Encoding numeric features

- **What:** For numeric features, which do not form natural groups, either bin them and encode the bins as categories, or fit a one-dimensional regression of the target on the feature and use its prediction as the encoding.
- **Cheap test (CPU, under 1 h):** Quantile-bin each numeric column into about 20 bins and apply technique 5. Alternatively, fit a univariate ridge or isotonic model per column inside the nested folds and use its prediction as a feature.
- **Evidence:**
  - L731-737: "когда у нас числовые признаки ... их будет очень много ... раскидать по бинам ... считать как будто бы это категориальной" (with numeric features there are many values, so bin them and treat them as categorical).
  - L744-748: "можно было просто построить линейный метод, регрессию поверх него, используя в качестве таргетом наш таргет" (one could fit a simple regression on the feature with the target, as an encoding for a numeric feature).

## 9. Greedy backward or forward feature pruning under cross-validation

- **What:** Rank features by importance and drop the least important one at a time, keeping the drop only if CV does not get worse. The speaker describes a backward variant, a forward re-add pass over the dropped features, and a shortlist that is then expanded with derived features.
- **Cheap test (CPU, under 1 h):** Use a fast model, such as ridge on a 20-column subset of the 112 columns, with the fixed folds. Run the greedy loop and log the CV score per step. Be aware that the speaker found the gain from hand-tweaked thresholds was noise-level, so require a minimum improvement.
- **Evidence:**
  - L874-882: "берём сначала все признаки ... отсортировать по важности ... убираем самый неважный ... смотрим улучшается или нет ... если улучшается, выкидываем, не улучшается, оставляем" (take all features, sort by importance, drop the least important, and keep the drop only if the score improves).
  - L888-893: "выкинули ... заново попробуем присоединять из выкинутых" (after dropping, retry adding back from the dropped set).
  - L894-899: "всё такими жадными алгоритмами ... потребовало существенное время вычислительное ... после каждого выкидывания хотелось бы ... быть уверенным ... потому что может быть просто шум" (all greedy algorithms; it needed substantial compute; after each removal you want confidence it is not just noise).

## 10. Brute-force interactions and a large blend of weak diverse models

- **What:** Generate pairwise, triple and quadruple interaction features, then train a grid of gradient boosting models over varying parameters, and average many of them. Many models were individually weak, but their sum still helped.
- **Cheap test (CPU, under 1 h):** Take the top 15 to 20 features by importance and generate all pairwise products or differences. Train a small grid of ridge or small GBM models and measure the blend gain over the best single model on the same folds. Stop if the blend gain is within the fold-to-fold noise.
- **Evidence:**
  - L619-634: "автоматический скрипт ... перебирает ... g-бусты с различными параметрами ... глубины деревьев ... числом соседей ... оригинальные признаки, потом двойные интеракции ... тройные ... четверные" (an automatic script enumerates many boosting models with varied parameters; original features, then pairwise, triple and quadruple interactions).
  - L660-665: "половина из них вообще никакого смысла не имели, но зато при объединении в сумме они что-то давали" (half of them made no sense, but summed together in the blend they contributed).
  - L619-621: the single top-ranking model he was asked about was a blend of 167 models.

## 11. Multi-level stacking with one fixed outer split and team-wide shared folds

- **What:** Fix a single outer fold split that everyone uses. Level-1 models produce out-of-fold predictions. Level-2 models consume them, and a final blend combines the levels. Inside the fixed outer folds, each member may choose their own inner splits for feature construction.
- **Cheap test (CPU, under 1 h):** Write one fold-assignment file and commit it with a hash. Build two or three level-1 models out of fold, then a level-2 linear blend, and check that the outer score is reproducible from the saved folds.
- **Evidence:**
  - L594-597: "построить одну пачку моделей на первом уровне, другую на втором уровне, и потом всё это аккуратно правильно смешать" (build one batch of models at level 1, another at level 2, then carefully blend all of it).
  - L808-812: "у нас изначально зафиксировали ... к-фолд валидацию ... для построения модели стекинга, а внутри ... мы могли как угодно внутри бить валидацию для построения фичи" (we first fixed the outer k-fold for the stacking model; inside it we could split however we liked to build features).
  - L866-869: "это были просто случайные ... фолды ... Казанова один раз случайно разбил фолды, выложил, и все стали этим пользоваться" (the folds were a single random split; one member split them once, uploaded them, and everyone used them).
  - L681-684: about 50 level-1 models, then 20 to 30 at level 2.

## 12. Feature engineering as the main lever

- **What:** The speaker says the main gain came from hand-built features, not from model choice. Standard learners on raw features gave weak results. Target encodings, interactions and recovered structure were the main source.
- **Cheap test (CPU, under 1 h):** Compare a standard learner on raw features with the same learner on raw features plus technique 5 or 6 encodings, under nested folds (technique 7). Report the gain and its spread across folds.
- **Evidence:**
  - L287-292: "мы пробовали все стандартные методы ... xgBoost, Random Forest ... любые линейные методы ... но всё это не давало какого-то сильно существенного качества" (we tried all standard methods; none gave a substantial improvement).
  - L690-692: "самое важное ... как я считаю ... фича-инжиниринг" (what I think mattered most was feature engineering).
  - L750-757: "все такие техники всегда отлично помогают ... один раз нужно только прогнать ... понять, где мы здесь переобучаемся" (these techniques always help; you only need to run them once to see where you overfit).

---

## What did NOT work or was judged not worth it

1. **Separate models per level of a key column.** The first split effectively separates on that column, which discards data from other groups (L326-334: "мы теряем некие данные там всех остальных этих групп" / "we lose some data from all the other groups").
2. **Hand-tuned pruning thresholds.** Adding or removing about five features by hand made results worse (L840-845). The speaker used a shared pruning script instead.
3. **Naive target encoding without nested folds.** It looks fine in CV because the leakage is hidden (technique 7).
4. **Plain count-difference encoding for small groups.** The raw difference fluctuates near zero when a group is small (L711-726), so it is less reliable than WoE (technique 6).
5. **Applying the derived features to all 130 raw columns instead of the 80 kept.** The speaker says someone else tried it and got about the same result (L949-962).
6. **Recomputing three-level stacking honestly.** The speaker decided it was not worth the cost at 167 models (L989-991). He also notes that at level 2 and 3, CV direction tracks the leaderboard but magnitude does not (L963-988).
7. **Denomination (scale) search that was never used.** The step-size search was done for differences that were ultimately not used, so it became a vestige (L1027-1034).

## Other notes

- The team kept full code with about 10 subfolders per member and hundreds of scripts (L1039-1042). The speaker says this made reproducibility hard to achieve, and that the sponsor paid unusually quickly before the call.
- The sponsor's questions focused on the inner and outer validation scheme, not on the features (L770-775).

---

## Ranked shortlist for our data (cheap, under 1 h, CPU only)

1. Technique 7: nested encoding check. This applies to any target-derived feature and is the highest-priority validity check.
2. Technique 1: exact linear dependency scan on the 112 columns, a cheap rank and R^2 pass.
3. Technique 2: lattice and step test on each column, to see whether any columns were discretized with noise. Skip continuous-looking columns.
4. Technique 4: distinct-count and eta^2 scan for grouping keys.
5. Technique 3: difference-based recovered ordering. Use only as a diagnostic, never as a causal input.
