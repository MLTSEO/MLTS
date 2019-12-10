# ML Problem Framing Worksheet

(This worksheet was transcribed into Markdown from the original provided by Kshitij Gautam. Neil Martinsen-Burrell also helped modify current doc.)

## Exercise 1: Start Clearly and Simply

**Write what you'd like the machine learned model to do.**

_We want the machine learned model to..._


**Example**: We want the machine learned model to predict how popular a video just
uploaded now will become in the future.

**Tips**: At this point, the statement can be qualitative, but make sure this
captures your real goal, not an indirect goal.

## Exercise 2: Your Ideal Outcome

**Your ML model is intended to produce some desirable outcome. What is this
outcome, independent of the model itself. Note that this outcome may be quite
different from how you assess the model and its quality.**

_Our ideal outcome is..._

**Example**: Our ideal outcome is to only transcode popular videos to minimize
server resource utilization.

**Example**: Our ideal outcome is to suggest videos that people find useful,
entertaining, and worth their time

**Tips**: You don't need to limit yourself to metrics for which your product
has already been optimizing. Instead, try to focus on the larger objective of
your product or service.

## Exercise 3: Your Success Metrics

**Write down your metrics for success and failyre with the ML system. The
failure metrics are important. Both metrics should be phrased independently of
the evaluation metrics of the model. Talk about the anticipated outcomes
instead.**

_Our success metrics are..._

_Our key results for the success metrics are..._

_Our ML model is deemed a failure if..._

**Example**: Our success metrics are CPU resource utilization. Our KR for the
success metric is to achieve a 35% reduced cost for transcoding. Our ML model
is a failure if the CPU resource cost reduction is less than the CPU costs for
training and serving the model.

**Example**: Our success metrics are the number of popular videos properly
predicted. Our KR for the success metric is to properly predict the top 95% 28
days after being uploaded. Our ML model is a failure if the number of videos
properly predicted is no better than current heuristics.

**Tips**: Are the metrics measurable? How will you measure them? (It's okay if
this is via a live experiment. Some metrics can't be measured offline.) When
are you able to measure them? (How long will it take to know whether your new
system is a success or failure?) Consider long-term engineering and
maintenance costs. Failure may not only be caused by non-achievement of the
success metric.

## Exercise 4: Your Output

**Write the output that you want your ML model to produce.**

_The output from our ML model will be..._

_It is defined as..._

**Example**: The output from our ML model will be one of the 3 classes of
videos (very popular, somewhat popular, not popular) defined as the top 3, 7,
or 90 percentile of watch time 28 days after uploading.

**Tips**: The output must be quantifiable with a definition that the model can
produce. Are your able to obtain example outputs to use for training data?
(How and from what source?) Your output examples may need to be engineered
(like above where watch time is turned into a percentile). If it is difficult
to obtain example outputs for training, you may need to reformulate your
problem.

## Exercise 5: Using the Output

**Write when your output must be obtained from the ML model and how it is used
in your product.**

_The output from the ML model will be made..._

_The output will be used for..._

**Example**: The prediction of a video's popularity will be made as soon as a
new video is uploaded. The output will be used for determining the transcoding
output for the video.

**Tips**: Consider how you will use the model output. Will it be presented to
a user in a UI? Consumed by subsequent business logic? Do you have latency
requirements? The latency of data from remote services might make them
infeasible to use. Remember the Oracle Test: if you always had the correct
answer, how would you use that in your product?


## Exercise 6: Your Heuristics
_

**Write how you would solve the problem if you didn't use ML. What heuristics
might you use?**

_If we didn't use ML, we would..._

**Example**: If we didn't use ML, we would assume new videos uploaded by
creators who had uploaded popular videos in the past will become popular
again.

**Tips**: Think about a scenario where you need to deliver the product
tomorrow and you can only hardcode the business logic. What would you do?

## Exercise 7a: Formulate Your Problem as an ML Problem

**Write down what you think is the best technical solution for your problem.**

_Our problem is best framed as:_
- _Binary classification_
- _Unidimensional Regression_
- _Multi-class, single-label classification_
- _Multi-class, multi-label classification_
- _Multidimensional regression_
- _Clustering (unsupervised)_
- _Other:_

_which predicts..._

**Example**: Our problem is best framed as 3-class, single label
classification which predicts whether a video will be in one of three classes
(very popular, somewhat popular, not popular) 28 days after being uploaded.

## Exercise 7b: Cast Your Prolem as a Simpler Problem

**Restate your problem as a binary classification or unidimensional
regression.**

_Our problem is best framed as:_
- _Binary classification_
- _Unidimensional regression_

**Example** We will predict whether upload videos will become very popular or
not. OR We will predict how popular an uploaded video will be in terms of the
number of views it will receive in a 28 day window.

## Exercise 8: Design your Data for the Model

**Write the data you want the ML model to use to make the predictions.**

_Input 1:_

_Input 2:_

_Input 3:_

**Example**: Input 1: Title, Input 2: Uploader, Input 3: Upload time, Input 4:
Uploaders recent videos

**Tips**: Only include information available at the time the prediction is
made. Each input can be a number or a list of numbers or strings. If your
input has a different structure, consider that is the best representation for
your data. (Split a list into two separate inputs? Flatten nested structures?)

## Exercise 9: Where the Data Comes From

**Write down where each input comes from. Assess how much work it will be to
develop a data pipeline to construct each column for one row.**

_Input 1:_

_Input 2:_

_Input 3:_

**Example**: Input 1: Title, part of VideoUploadEvent record, Input 2:
Uploader, same, Input 3: Upload time, same, Input 4: Recent videos, list from
a separate system.

**Tips**: When does the example output become available for training purposes?
Make sure all your inputs are available at serving tie in exactly the format
you specified.

## Exercise 10: Easily Obtained Inputs

**Among the inputs you listed in Exercise 8, pick 1-3 that are easy to obtain
and would produce a reasonable initial outcome.**

_Input 1:_

_Input 2:_

**Tips**: For your heuristics, what inputs would be useful for those
heuristics? Focus on inputs that can be obtained from a single system with a
simple pipeline. Start with the minimum possible infrastructure.
