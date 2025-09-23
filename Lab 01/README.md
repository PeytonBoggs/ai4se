## Lab 01

### Methodology

Since utilizing LLMs to write the Java method mining script worked so well, I again tried that method to complete the ngram model code. I input the n-gram-recommender.md file along with a description of the dataset in the csv file into Claude, and it generated the ngram_recommender.py file. Surprisingly, this code worked on the first try, and did not need modifying like the LLM outputs of Lab 01 did. The script loads the dataset, trains 3-gram, 5-gram, and 7-gram models, evaluates the models based on perplexity, and selects the best performing model. It also generates 1000+ predictions and saves results to the ngram_predictions.jsonl file.

### Output

After training the three 3-gram, 5-gram, and 7-gram models, the code evaluates them on perplexity. The 3-gram model was given a perplexity of 2253.74, the 5-gram model recieved a perplexity of 10183.18, and finally the 7-gram got a perplexity of 20092.93. This clearly shows that the 3-gram was the best model trained.

On top of this, some example predictions were output. Here is one:

```
Input context: public String
Predictions:
  'toString' → 0.003
  'getAsText' → 0.001
  '[' → 0.001
  'print' → 0.001
  'getName' → 0.001
Sampled completion: toString ( ) ; Object key , "123" ) ;
```

This appears to be working correctly, as the tokens 'public String toString(); Object key, "123";' seem reasonable as code. I was surprised that the share of the winning token 'toString' only had a probability of 0.003, but I suppose that with such a large dataset that is a relatively large probability.

### How to Run

To run, just input the dataset in a csv file to the script:

```
python3 ngram_recommender.py java_methods.csv
```

You can also add the following flags:

```
--n-values //n-gram sizes to evaluate, default is [3, 5, 7]
--samples //number of test samples for prediction, default is 1000
--output //output file, default is ngram_predictions.jsonl
```
