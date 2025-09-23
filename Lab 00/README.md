## Lab-00: Peyton Boggs

### Methodology

For this lab, I was inspired by how well the GPT5 model generated the GitHub scraper earlier and class, and while I knew that I could spend some time mining methods from GitHub and tweaking regex patterns to fit them into the proper fields myself, I wanted to see how different models might do the task. So, I gave the original mining-java-methods.md lab file along with some guidance first to Perplexity AI, and it spit out a file that I worked with for quite a while but never got to function properly. Then, I turned to Claude and asked it to modify the Perplexity code. To my surprise, it generated an entire website that looked like it did exactly what I needed in just a few seconds.

With Claude, I published the webside and it is availiable here: https://claude.ai/public/artifacts/774cd890-b54a-4326-976c-e4a77d43cf73

This website appears to give reccommended repos and easy buttons to scrape and download the cleaned data. This seemed incredibly easy and blew me away at first.

However, after looking into the data, I soon realized that it was all faked. I should have known that it was too good to be true, but looking into the code, GitHub's API is never used and the methods that it "extracts" are basically all the same.

Finally, I turned to ChatGPT, which similarly to Perplexity gave me a file that seemed promising but needed some work. After about an hour of tinkering with it, I was able to get the script to scrape the needed methods off of GitHub.

### Description

My file `mining-java-methods.py` has hardcoded popular GitHub repos with a lot of Java methods. It uses my GitHub token to go through each one as needed (untill 25,000 methods are scraped) and used javalang to parse each method into it's proper fields. Importantly, it cleans the data through only using methods with a set minimum/maximum lines, and deduplicates the data through the method's unique hashes. The generated csv file, `java_methods.csv`, has exactly 25,000 methods, and the average method has 8.35 lines. Incidentally, all methods came from the `spring-projects/spring-framework` repo.

## How to Run

To run the program, execute the following:

```
pip install javalang requests tqdm python-dotenv
python3 mining-java-methods.py --out-file ./java_methods.csv
```

You can also use the following flags:

```
--token //specify GitHub token, default is your os env's "GITHUB_TOKEN"
--target-methods //number of methods to scrape, default 25000
--train-count //number of methods in training set, default 20000 (the rest will be in eval set)
```