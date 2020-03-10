
# Introduction

Jeopardy is a popular TV show in the US where participants answer questions to win money. It's been running for a few decades, and is a major force in popular culture.

Let's say you want to compete on Jeopardy, and you're looking for any edge you can get to win. In this project, I'll work with a dataset of Jeopardy questions to figure out some patterns in the questions that could help you win.

The dataset is named jeopardy.csv, and contains 20000 rows from the beginning of a full dataset of Jeopardy questions, which you can download here: https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/

Here are the first few rows of the dataset.




```python
import pandas as pd

jeopardy = pd.read_csv("jeopardy.csv")
jeopardy.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, each row in the dataset represents a single question on a single episode of Jeopardy. Here are explanations of each column:

* **Show Number** indicates the Jeopardy episode number of the show this question was in.
* **Air Date** represents the date the episode aired.
* **Round** is the round of Jeopardy that the question was asked in. Jeopardy has several rounds as each episode progresses.
* **Category** refers to the category of the question.
* **Value** is the number of dollars answering the question correctly is worth.


```python
#Removing leading and tailing white spaces at the column names.
jeopardy.columns = jeopardy.columns.map(lambda x: x.strip())
```

# Normalizing Strings (Question and Answer)

Before you can start doing analysis on the Jeopardy questions, I need to first normalize the question and answer column. To do that, I simply remove punctations and lowercased all characters.


```python
import string

def normalize_string(text):
    lowercase_text = text.lower()
    character_list = [c for c in lowercase_text if c not in string.punctuation]
    normalize_text = ''.join(character_list)

    return normalize_text

jeopardy['clean_question'] = jeopardy['Question'].apply(normalize_string)
jeopardy['clean_answer']   = jeopardy['Answer'].apply(normalize_string)
```


```python
#Preview the first 5 rows to check if normalization works.
jeopardy.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
      <th>clean_question</th>
      <th>clean_answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
      <td>for the last 8 years of his life galileo was u...</td>
      <td>copernicus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
      <td>no 2 1912 olympian football star at carlisle i...</td>
      <td>jim thorpe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
      <td>the city of yuma in this state has a record av...</td>
      <td>arizona</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
      <td>in 1963 live on the art linkletter show this c...</td>
      <td>mcdonalds</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
      <td>signer of the dec of indep framer of the const...</td>
      <td>john adams</td>
    </tr>
  </tbody>
</table>
</div>



# Normalize Integers and Dates (Value and Air Date)

Now that I've normalized the text columns, there are also some other columns to normalize.

The Value column should also be numeric, to allow you to manipulate it more easily. I'll need to remove the dollar sign from the beginning of each value and convert the column from text to numeric.

The Air Date column should also be a datetime, not a string, to enable you to work with it more easily.


```python
def normalize_integer(string_integer):
    if string_integer == 'None':
        return 0
    else:
        character_list = [c for c in string_integer if c not in string.punctuation]
        integer = int(''.join(character_list))

        return integer

jeopardy['clean_value'] = jeopardy['Value'].apply(normalize_integer)
jeopardy['Air Date'] = pd.to_datetime(jeopardy['Air Date'])
```

# Answers in Questions

In order to figure out whether to study past questions, study general knowledge, or not study it all, it would be helpful to figure out two things:

* How often the answer is deducible from the question?
* How often new questions are repeats of older questions?

We can answer the second question by seeing how often complex words (> 6 characters) reoccur. We can also answer the first question by seeing how many times words in the answer also occur in the question. We'll work on the first question now, and come back to the second.


```python
#proportion match sees what proportion of the answer appears
#in the question
def proportion_match(jeopardy_row):
    split_answer   = jeopardy_row['clean_answer'].split(" ")
    split_question = jeopardy_row['clean_question'].split(" ")
    length_split_answer = len(split_answer)

    match_count = 0
    if 'the' in split_answer: split_answer.remove('the')
    if length_split_answer == 0: return 0

    for word in split_answer:
        if word in split_question:
            match_count += 1

    return match_count / length_split_answer

#use proportion match on every row on the dataset and compute
#the average proportion of the words in the answer that appear
#in the question
jeopardy['answer_in_question'] = jeopardy.apply(proportion_match, axis = 1)
jeopardy['answer_in_question'].mean()
```




    0.05720910976173767



For the first question, it appears that on average the answer appears in the question about 6% of the time. This implies that we cannot simply use the above strategy if we want to win in Jeopardy.

# Recycled Questions

Let's say you want to investigate how often new questions are repeats of older ones. To do this, you can:

* Sort jeopardy in order of ascending air date.
* Maintain a set called **terms_used** that will be empty initially.
* Iterate through each row of jeopardy.
* Split clean_question into words, remove any word shorter than 6 characters, and check if each word occurs in terms_used.
 * If it does, increment a counter.
 * Add each word to **terms_used**.

This will enable you to check if the terms in questions have been used previously or not. Only looking at words greater than 6 characters enables you to filter out words like the and than, which are commonly used, but don't tell you a lot about a question.


```python
questions_overlap = []
terms_used        = set()
sorted_jeopardy   = jeopardy.sort_values('Air Date')

for (_, row) in sorted_jeopardy.iterrows():
    split_question  = row['clean_question'].split(" ")
    split_question  = [word for word in split_question if len(word) >= 6]
    length_question = len(split_question)

    match_count = 0
    for word in split_question:
        if word in terms_used: match_count += 1
        terms_used.add(word)

    if length_question > 0: match_count /= length_question
    questions_overlap.append(match_count)

jeopardy['questions_overlap'] = questions_overlap
jeopardy['questions_overlap'].mean()
```




    0.6889055316620302



68% of the data set contains questions that are recycled from past questions. This means that we can study recycled questions and hopefully win in Jeopardy.

# Low Value vs. High Value Questions

Let's say you only want to study questions that pertain to high value questions instead of low value questions. This will help you earn more money when you're on Jeopardy.

You can actually figure out which terms correspond to high-value questions using a chi-squared test. You'll first need to narrow down the questions into two categories:

* Low value -- Any row where Value is less than 800.
* High value -- Any row where Value is greater than 800.

You'll then be able to loop through each of the terms from the last screen, terms_used, and:

* Find the number of low value questions the word occurs in.
* Find the number of high value questions the word occurs in.
* Find the percentage of questions the word occurs in.
* Based on the percentage of questions the word occurs in, find expected counts.
* Compute the chi squared value based on the expected counts and the observed counts for high and low value questions.

You can then find the words with the biggest differences in usage between high and low value questions, by selecting the words with the highest associated chi-squared values. Doing this for all of the words would take a very long time, so we'll just do it for a small sample now.


```python
def is_high_value(row):
    if row['clean_value'] > 800: value = 1
    else: value = 0

    return value

jeopardy['high_value'] = jeopardy.apply(is_high_value, axis = 1)

def high_low(word):
    low_count  = 0
    high_count = 0

    for (_, row) in jeopardy.iterrows():
        split_question = row['clean_question'].split(" ")
        if word in split_question:
            if row['high_value'] == 1: high_count += 1
            else: low_count += 1

    return high_count, low_count

observed_expected = []
comparison_terms  = list(terms_used)[0:5]
for term in comparison_terms:
    high, low = high_low(term)
    observed_expected.append([high, low])
```

# Applying the Chi Square Test

Now that you've found the observed counts for a few terms, you can compute the expected counts and the chi-squared value.


```python
from scipy.stats import chisquare

high_value_count = len(jeopardy[jeopardy['high_value'] == 1])
low_value_count  = len(jeopardy[jeopardy['high_value'] == 0])

chi_squared = []
for high_low in observed_expected:
    observed_high = high_low[0]
    observed_low  = high_low[1]

    total      = high_low[0] + high_low[1]
    total_prop = total / len(jeopardy)

    expected_high = total_prop * high_value_count
    expected_low  = total_prop * low_value_count

    chi_p = chisquare([observed_high, observed_low],
                      f_exp = [expected_high, expected_low])

    chi_squared.append(chi_p)

chi_squared
```




    [Power_divergenceResult(statistic=0.401962846126884, pvalue=0.5260772985705469),
     Power_divergenceResult(statistic=0.401962846126884, pvalue=0.5260772985705469),
     Power_divergenceResult(statistic=0.803925692253768, pvalue=0.3699222378079571),
     Power_divergenceResult(statistic=0.401962846126884, pvalue=0.5260772985705469),
     Power_divergenceResult(statistic=0.803925692253768, pvalue=0.3699222378079571)]



Based on the above test, it implies that there were no differences between the number of high value questions and low value questions.
