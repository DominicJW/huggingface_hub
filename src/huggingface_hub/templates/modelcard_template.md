---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for AnyLengthAlbert

AnyLengthAlbert is designed for sentence pair classification tasks. It is built upon the base of Albert-base-v2. The model expects inputs: input_ids: (Batch Size x Max Number of Chunks x Max Length of Chunk), attention_mask: (Batch Size x Max Number of Chunks x Max Length of Chunk), number_of_chunks: (Batch Size)
The model does a forward pass feeding each chunk to albert, and then pools the CLS outputs, (sum(MaxPool, AvgPool)) and passes that to a single dense layer, generating a single output. The output should then be passed to through a sigmoid function  to generate probabilities.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

{{ model_description | default("", true) }}

- **Developed by:** Dominic Johnston-Whiteley
- **Funded by [optional]:** Dominic Johnston-Whiteley
- **Model type:** Transformer
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Finetuned from model [optional]:** bert-base-uncased

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/DominicJW/AnyLengthBert
- **Demo [optional]:** https://colab.research.google.com/drive/1xRCR1EA1C8iWvkZ1dFdj0t6f7qgH7HWU?usp=sharing 

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Detecting if one sentence is evidence for a claim made by anotehr
### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Fact checking (See below)

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
Not to be used for any real life application. 
The scope it can be used in is within this coursework. Though the archetecture could inspire other attempts to handle longer input sequences than a single transformer encoder model is able to. 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

I achieved higgher performing results with vanilla finetuning of Albert. Training data has not been vetted by me personally, and I have seen not seeked any information about how the data was collected, or cleaned. 
A key limitation is the need for complex pre-proccessing of the text.
A limitation I found in the training data is that most of the 'claims' could be seen as expressing an opinion. So a more accurate description of the models capabilities is that it has been trained to detect whether the 'evidence' justifies the opinion. 

For example: (not taken from dataset)
We should Elect Trump : When Trump was in office, he came out of many climate treaties, this boosted the american economy but at the cost of the environment : True | False
with a pair of sentences like this, 
the label True or False is entirely down to the opinions of the labeller, whether they see the climate or economy as more important.
This example, though not taken from the dataset highlights a key potential issue with claims which have conflicting evidence.

(taken from the dev dataset)
Claim: Evidence: Label
We should legalize same sex marriage :	A June 2006 TNS-Sofres poll found that 45% of respondents supported same-sex marriage, with 51% opposed. :	1

The evidence that a poll found slightly more people opposed same sex marriage than supported it or were neuteral makes me question the critereon the labeller used to distinguish whether something the evidence supports is neutral or is against the claim. 


We should increase internet censorship :	According to the report, few countries demonstrated any gains in Internet freedom, and the improvements that were recorded reflected less vigorous application of existing controls rather than new steps taken by governments to actively increase Internet freedom.:	1

In this example the evidence shows governments are doing X, the label implies that if governments are doing X then we should do X. Clearly the labels have not been well thought through.

I did not have to delve deep into the dataset to find these examples. It took me around 30 seconds to find each one just by skimming. This shows that the dataset is likely riddled with similar issues. Due to the severe and clear issues, I assume this was some kind of test by our lecturer, to see if we would consider the ethical issues underlying the dataset. This is very important and relevant today, with the advent of automated fact checkers censoring social media during the pandemic. And indeed this dataset highlights one of the ways such systems can get things wrong.

Fact checking gone wrong: https://www.bmj.com/content/376/bmj.o95#ref-8 

Note: Evidence detection is not the same task as fact checking, but the closely related task, natural language inference/ textual entailment can be used in the fact checking pipeline
For example: https://aclanthology.org/2021.findings-acl.217




### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->
The model does not have good enough performance to use in any kind of production setting. It is merely for a coursework, and an attempt to use sequences longer than 512 with a bert based transformer, as opposed to nievly truncating the input sentence.

As discussed above, there are many disputable labels on the training data. 
Do not use!

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Supplied by The University of Manchester, for this coursework. 
Training data was a csv file consisting of three columns: Claim, Evidence, label. And around 23 thousand samples, with a class imbalance of 75% false, 25% true. 

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]


Raw sentence pairs must be proccessed by using the associated CustomDataset, and using the custom collate function in the dataloader to create padding chunks across each batch.

In training, upsampling was used to correct the class imbalance.

#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

bert_model = "bert-base-uncased"
freeze_bert = False
iters_to_accumulate = 1
lr = 2e-5  
epochs = 4  
gradient_clip_val = 5.0
bs = 32 
num_training_steps = len(train_loader) * epochs
optimizer = AdamW, default params except lr
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
  num_warmup_steps=500,
  num_training_steps=num_training_steps)
Early stop at end of epoch 2!.

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}
max length 64: 
max length 128: 1min30s
max length 256: 
max length 512: 

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Supplied by University of Manchester

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

No disaggreagation used in testing.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"MCC: {mcc}")
print(f"Accuracy: {accuracy}")

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** L4 GPU
- **Hours used:** To train once: 10 mins (at max length 128), To develop: dozens of trials at 10-20 mins each
- **Cloud Provider:** Google Colab
- **Compute Region:** Europe (likely)
- **Carbon Emitted:** 0.63 kg CO2 eq.  (All offset by google, calculated using above link, said V100 as most comparable to L4, and L4 not available in drop down)


## Model Card Authors [optional]
Dominic Johnston-Whiteley

## Model Card Contact
dominic.johnstonwhiteley@student.manchester.ac.uk
