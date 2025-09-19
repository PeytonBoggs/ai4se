## Lab-00: Peyton Boggs

For this lab, I first wanted to modify my GitHub scraper from the MSR Lab so that is would scrape only Java methods. 
Because GPT5 worked so well for that lab, I gave my solution to Perplexity along with the example CSV line and asked it to modify the scraper to extract only that information. 
Perplexity gave me "mining-java-methods.py", which I tinkered with for about an hour until I gave up because I still could not successfully scrape a method.

Then, I turned to Claude and asked it to modify the Perplexity code. 
This worked well, but I was still having trouble with the methods, and figured it was because the repositories I was using did not have clean Java methods.
I realized that Claude needed more context, so I gave it the entire "mining_java_methods.md" file, and to my surprise, it generated an entire website that did exactly what I needed in just a few seconds.

With Claude, I published the webside and it is availiable here: https://claude.ai/public/artifacts/774cd890-b54a-4326-976c-e4a77d43cf73

This website gave reccommended repos and easy buttons to scrape and download the cleaned data. This was incredibly easy and blew me away after an hour of parsing through the details.

The training and evaluation csv files were generated through this website, as well as the "dataset_description.txt" file, which was generated because that was part of the lab's deliverables.