# How I Approached the Santander Customer Transaction Prediction Competition

**Speaker: Vasily Ryazanov**

Hi, everyone. My name is Vasily Ryazanov, and I'd like to tell you how I approached this competition. I finished 21st out of almost 9,000 teams, which earned me a gold medal. I competed on my own. Rather than just showing you my final model, I want to walk you through the process: what I tried, what I discovered, and how my thinking changed along the way. So this will be a bit of a story.

## Previous Santander competitions

First, a little background on Santander's competitions on Kaggle. This was their third competition, and each of them had something unusual about it.

I don't remember the exact cause of the problem in the first one, but there was a massive leaderboard shake-up, with some teams moving thousands of positions in either direction. It was probably caused by a time dependency in the data. There was a popular Kaggle kernel that a lot of people used, and many of them ended up dropping dramatically in the rankings. On the slide, you can see top-ranking teams losing several thousand places, and some falling even further.

In the second competition, there was an outright data leak. If you looked carefully at certain features, you could see that the data was essentially a time series—and one that could be traced in both directions. A well-known Kaggle participant discovered this and published the finding. The values followed one another in a very clear sequence.

Now let's move on to the third competition.

## The task and the data

The competition was called *Santander Customer Transaction Prediction*. The organizers gave us an anonymized dataset and almost no information about the features. They said that the data had the same structure as the data they used to predict real customer transactions. The task was to predict whether a customer would make a transaction of a particular type.

It was a binary classification problem, with a roughly 90/10 class split, evaluated using ROC AUC. There were 200 features, 200,000 rows in the training set, and 200,000 rows in the test set. All the feature values appeared to be rounded to four decimal places.

Many features looked approximately normally distributed, although some had outliers or other anomalies. Their means, variances, and ranges differed, but overall the values fell roughly between -90 and 90. There were no missing values. The number of distinct values in a feature was also related, to some extent, to its variance, although nothing particularly striking stood out at first.

Here are some histograms of individual features, split by target class and compared with the test set. Pairwise plots looked much the same. Some distributions were very close to normal; others had outliers, irregularities, or noticeable spikes at particular values. Beyond this, we had almost no information, so everyone on the forum was trying to guess how the dataset had been constructed.

## My first baseline

My first pipeline was probably very similar to what many other participants tried, judging by the forum discussions. I started with the raw features, then generated combinations: sums, differences, products, and so on. What else can you do when the features are anonymous?

I also calculated row-wise statistics and tried target-mean encoding for features with relatively few unique values. I experimented with blending logistic regression, k-nearest neighbors—which actually worked reasonably well—and boosting models. But the overall ROC AUC on the leaderboard was still disappointing. It wasn't even reaching 0.900.

With hyperparameter tuning—either following recommendations from the forum or running my own optimization—it was possible to get to around 0.900. But that was still a fairly mediocre result, somewhere in the middle of the leaderboard.

About a month or a month and a half before the competition ended, the leaderboard looked very strange. A handful of teams were far ahead, with scores of about 0.904, 0.910, or even 0.912. Another five to ten teams were around 0.901, while thousands of teams were stuck at 0.900 or 0.899.

It was obvious that there was something hidden in the data. People began calling it the *magic*. Some called it a leak; others preferred the word magic, and there were even debates about which term was more appropriate. But everyone understood that there was something important to discover, and people started looking for it.

## Searching for the “magic”

One of the first unusual things people noticed was that you could shuffle each feature independently within each target class in the training data. In other words, shuffle the values within target = 0 and within target = 1 separately, preserving each feature's class-conditional distribution but destroying the relationships between features. You could train on the resulting dataset and get essentially the same leaderboard score—or sometimes even a slightly better one.

That was strange, but it didn't yet lead to a major breakthrough. It suggested that the features were nearly independent: correlations between them were close to zero.

People tried PCA, t-SNE, different kinds of autoencoders and neural networks, and searches for categorical features hidden among columns with few distinct values. Some tried to find a time-series structure, perhaps by identifying Unix timestamps and converting them into weeks or other time units. None of those ideas really worked.

People also tried data augmentation based on the shuffling trick: they generated additional training rows by independently permuting features within each target class, then added those synthetic rows to the original training set. I experimented with all of this too. It took a lot of time, and at that stage the results weren't particularly useful.

## The first important clue: duplicate values

Here's how I eventually started to find the magic. The first thing that caught my attention was the number of repeated values in each row.

For each feature in a row, I checked whether its value also appeared in that same feature in some other row. In other words, was the value unique or not? I then counted the number of non-unique values per row and looked at how this count behaved across the dataset. A rolling average made a striking difference between the training and test sets immediately visible.

When I looked at the distribution more closely, I found something surprising. The 200,000 training rows and 100,000 of the test rows had an approximately normal distribution of duplicate counts, with a mean around 160 out of 200 features. But there were exactly 100,000 other test rows for which the count was 200. Every single feature value in those rows appeared somewhere else: not one of their values was unique.

That was a very unusual pattern, so I began investigating it. I also noticed that those 100,000 test rows didn't seem to contribute to the public leaderboard score. At that point, I suspected they belonged to the private evaluation set, although I hadn't yet understood what they really were.

Another observation was that every value in those rows had been taken from somewhere: the values were all present in other test rows. In effect, one half of the test set contained unique values, while the other half was assembled from values occurring elsewhere in the test set.

I returned to my feature-engineering pipeline and introduced duplicate-related statistics. For the training rows, I counted whether a value was repeated in the training data; for the test rows, I also took into account whether that value appeared in training. The resulting distributions were similar, and these features raised my score to around 0.900–0.901. At that moment, a score like that was enough to be in the gold-medal range, somewhere around 10th to 12th place.

I experimented further by concatenating different feature groups. I used raw features; binary indicators showing whether each value was duplicated; and transformed features where a unique value was replaced by that feature's mean, while a non-unique value was left unchanged.

## Identifying fake rows in the test set

Around that time, a very important Kaggle kernel and forum discussion appeared. Someone had independently discovered essentially the same pattern, but analyzed it more deeply than I initially had.

Like me, the author found 100,000 test rows with no unique feature values. But they also discovered that all the values used to construct those rows could be traced back to just 50,000 other test rows—not to all 100,000 rows in the other half of the test set.

The idea was to inspect each value from the rows we could now call *fake* rows, then identify which rows in the other part of the test set contained those values. That let the author distinguish rows that had apparently served as sources, or *generators*, for the fake observations from rows that hadn't.

The 100,000 apparently real test rows split neatly into two groups of 50,000. The 50,000 generator rows appeared to belong to the public leaderboard subset, while the remaining 50,000 belonged to the private subset. The 100,000 fake rows appeared not to be used in either score. This matched the organizers' statement that the public and private leaderboard evaluations were split roughly 50/50. It also explained why the fake rows looked so different from the rest of the test data.

At that point, we became much more confident that these really were artificially generated test rows and that they wouldn't be used to compute the final score.

I reworked my model using this insight, and my leaderboard ROC AUC jumped to about 0.922. Instead of computing frequency statistics only from the training set, I added the 100,000 real test rows and recalculated the value counts and related features, excluding the fake rows.

Of course, the other competitors noticed this development too, and many of them rapidly improved their scores. The group of participants scoring above 0.901 grew much larger.

## Feature engineering that worked

One transformation I found particularly useful was based on value frequencies, or value counts.

I calculated the frequency of each feature value and moved the original value toward the mean of that feature, with the amount of movement depending on how frequently the value occurred. I chose the weighting function experimentally. It was approximately a fourth-root transformation of the normalized frequency.

More precisely, I scaled the count values to the interval from 0 to 1, took the fourth root, and used the resulting coefficient to interpolate between the feature mean and the original value. If a value was among the most frequent, I kept it almost unchanged. If it was extremely rare, I replaced it almost entirely with the feature mean.

My final feature set included the raw columns, binary duplicate indicators, value-count statistics, these frequency-weighted transformations, and threshold-based replacements in which sufficiently rare values were replaced with the mean. I also used several row-wise statistics, mostly calculated from the frequencies rather than from the raw feature values.

That produced a lot of features—roughly 1,500, possibly more. Keeping all of them was expensive in terms of memory. So I used SHAP to estimate feature importance and removed features with zero or near-zero SHAP importance. This both reduced memory usage and noticeably improved the score.

I also used data augmentation. This was the same trick I described earlier: shuffle feature columns independently within each target class and add the shuffled rows as new training samples. Depending on the experiment, I expanded the training set by a factor of two or three. I then blended models trained on these augmented datasets.

I blended LightGBM and CatBoost models as well, using rank-based blending because ROC AUC was the evaluation metric. It worked fairly well.

Among the gradient-boosting libraries, I liked LightGBM the most. It was genuinely fast and worked very well. CatBoost was decent, but slower and produced similar scores. Perhaps someone can teach me to tune CatBoost better. XGBoost was quite slow for me, and a couple of times it even crashed my kernel, so I eventually stopped using it.

## Computing resources

In terms of hardware, I had a home workstation where I trained most of the main models. I also had a laptop for hyperparameter tuning, examining distributions, and other exploratory work, so I didn't overload the main machine. Later, I rented a memory-optimized Amazon EC2 instance to run routine experiments that didn't require much supervision.

## What I was missing, and how I could have finished higher

Let's talk about what else could have been done to place higher, and what I personally lacked.

There's a Kaggle forum thread collecting a large number of gold-medal solutions—probably ten to fifteen of them. I competed alone, and it was mentally quite difficult, especially toward the end. At one point I was holding second to fourth place on the leaderboard, but during the final week everyone suddenly became much more active. I was dropping several places almost every day, eventually finishing 21st. That was pretty nerve-racking.

More than anything, I think I ran out of ideas. The whole competition felt like a mathematical-olympiad puzzle: you had to concentrate on abstract, anonymous data and discover some hidden structure.

One thing I failed to do was find an effective way to ensemble my solution with a neural network. If you look at some of the top solutions, simply averaging predictions with neural-network models brought significant gains. Some of those solutions reached ROC AUC scores around 0.925.

There was also a particularly interesting trick used by the fourth-place team, which included the Russian Kagglers known as ZFTurbo and Evgeny Patekha, together with their teammate alijs.

The trick was to *unpivot* the dataset. Instead of treating the 200 anonymous feature columns as completely different variables, you could view their values as instances of a single numerical variable, with an additional categorical feature indicating which original column each value came from.

You could also add the value frequency, as that team did. The original table with 200,000 training rows and 200 features would then turn into a long table with 40 million rows. Each original feature value became its own training example, while retaining the original row's target and the identity of the original feature—`var_0`, `var_1`, and so on.

If you trained a binary classifier directly on that long table, the ROC AUC was unimpressive, around 0.53. But at least that was higher than 0.5. The key was to group the predictions back by the original customer and aggregate them. According to the team, this approach could produce a score around 0.925, which was an excellent result. My own final result was also around 0.925, but it required various models and blends; their approach was a powerful standalone idea.

## What were these data, and why did the magic work?

A few thoughts on what the dataset actually represented. In the end, I don't think anyone fully understood why the magic worked. Santander didn't really explain it. By the time the competition was over, people were less motivated to keep investigating.

It was clear that the data were either synthetic or had been generated or transformed in some highly unusual way.

One possibility is that the organizers were responding to privacy concerns in Europe. Analyzing actual individuals' data is increasingly sensitive, but perhaps you can work with artificial data that resemble real customer data. If a model learns useful patterns from data that look like real customer records, maybe it can later be applied to real records. That's one possible explanation—assuming Santander could construct synthetic data that preserved the relevant patterns.

Another possibility is simply that Santander was very protective of its data and tried to obfuscate it as thoroughly as possible.

Why did the so-called magic—frequency statistics, removing fake test observations, and related features—work so well? We never got a definitive answer. My guess is that it had to do with the synthetic or heavily transformed nature of the dataset. Perhaps the values were independently generated from various distributions or transformed and then rounded to four decimal places.

In such a setting, boosting models can learn not only from splits on the numerical values of the features but also from splits based on how frequently those values occur. That additional signal might be what improved the models. So if you encounter a dataset with similar characteristics, value-count features are definitely worth trying.

## Practical advice from the competition

A few things made the process easier for me. During the last week or two, I was spending 10 to 15 hours a day on the competition, and I'd already spent about a month working on it at a somewhat less intense pace.

First, always make Git commits so you don't lose track of your experiments and decisions. You can even write a helper function to commit automatically, so you don't have to dig through old experiments later. Ideally, when you achieve a good score, you should be able to identify the exact code and model that produced it.

Second, send your logs to Telegram. It's very convenient. There are tools that can forward progress bars, training updates, plots, and other experiment information there, so you can keep an eye on long-running jobs.

Third, I recommend SHAP for feature-importance analysis. It ranks features using a game-theoretic approach, and it can also show how different ranges of a feature contribute toward predicting target class 0 or 1. This can help both with feature selection and with understanding what your features are doing. In this competition, SHAP wasn't especially helpful for understanding the anonymous features, but it worked well for filtering them.

I'd also recommend recording experiment times and scores in your commit messages or experiment logs, so you can compare runs and trace your results later.

Another useful practice is to study the strongest competitors. If someone is near the top of the leaderboard, look through their public kernels and forum discussions, sorted by date. Maybe they've shared something useful about the current competition. Maybe they discovered a leak in an earlier competition. Understanding what kinds of patterns they've found before can suggest what to look for now.

I also checked GitHub and searched for publicly available competition-related work, including sorting results by date. I wouldn't say I found anything particularly valuable there, but it's another possible avenue of investigation.

Anyway, that was my journey through the competition. Any questions?

## Questions and answers

**Audience member:** Thank you for the presentation. It was very interesting. Could you explain the trick where you expanded the dataset to tens of millions of rows? What exactly did you do with it afterward?

**Vasily:** Sure. That was the fourth-place team with ZFTurbo and Evgeny Patekha. You can ask them directly, of course, but here's how I understood their approach.

You start with 200 features describing one observation. Instead of keeping all 200 values in one row, you create 200 separate rows. Each new row contains the value of one original feature, the target, and an additional categorical feature identifying which variable it came from—`var_0`, `var_1`, and so forth.

They also added the frequency of the value, which was one of the key features. So each row of the expanded table effectively contained the raw feature value, its count or frequency, the original feature identifier, and the target.

The dataset grows in proportion to the number of features. You train a model on this expanded table. The ROC AUC of the individual predictions isn't very good, but once you combine those predictions and aggregate them back to the original observation level, you get a strong score.

In effect, this changes the nature of the representation completely. And it worked surprisingly well for them.

**Another audience member:** Thank you for the talk. Do you think Santander got any real benefit from this competition? If so, what was it?

**Vasily:** That's still a mystery to me, actually. The forum went quiet after the competition, because people weren't particularly interested in digging into the dataset anymore.

The explanation I find most plausible is the one I mentioned earlier: there may come a time when analyzing real individuals' data becomes much more restricted, but analyzing artificial data that resemble those individuals is still possible. If you teach data scientists to work with similar, artificially generated datasets—with values drawn from different distributions—perhaps you're preparing them for that future. Maybe Santander was testing or preparing for exactly that scenario.

Otherwise, it's difficult to see what practical value Santander gained. The dataset was so strange, and nobody managed to establish a convincing connection to actual real-world data.

It's also possible that the target distribution itself was artificially altered. For example, maybe the real positive class originally represented only 1% of observations, rather than 10%, and the organizers expanded it somehow. That could also help explain why frequency-based statistics produced such a large improvement in the score.
