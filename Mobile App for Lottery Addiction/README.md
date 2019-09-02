
# Introduction

Many people start playing the lottery for fun, but for some this activity turns into a habit which eventually escalates into addiction. Like other compulsive gamblers, lottery addicts soon begin spending from their savings and loans, they start to accumulate debts, and eventually engage in desperate behaviors like theft.

A medical institute that aims to prevent and treat gambling addictions wants to build a dedicated mobile app to help lottery addicts better estimate their chances of winning. The institute has a team of engineers that will build the app, but they need us to create the logical core of the app and calculate probabilities.

For the first version of the app, they want us to focus on the 6/49 lottery and build functions that enable users to answer questions like:

* What is the probability of winning the big prize with a single ticket?
* What is the probability of winning the big prize if we play 40 different tickets (or any other number)?
* What is the probability of having at least five (or four, or three, or two) winning numbers on a single ticket?

The institute also wants us to consider historical data coming from the national 6/49 lottery game in Canada. The data set has data for 3,665 drawings, dating from 1982 to 2018 (we'll come back to this).

# Core Functions


```python
# n and k are non-negative integers that satisfies the 
# inequality n >= k

def factorial(n):
    if n <= 1: return 1
    else: return n * factorial(n - 1)

def combinations(n, k):
    numerator   = factorial(n)
    denominator = factorial(n - k) * factorial(k)
    
    return numerator / denominator
```

# One-ticket Probability

In the 6/49 lottery, six numbers are drawn from a set of 49 numbers that range from 1 to 49. A player wins the big prize if the six numbers on their tickets match all the six numbers drawn. If a player has a ticket with the numbers {13, 22, 24, 27, 42, 44}, he only wins the big prize if the numbers drawn are {13, 22, 24, 27, 42, 44}. If only one number differs, he doesn't win.

For the first version of the app, we want players to be able to calculate the probability of winning the big prize with the various numbers they play on a single ticket (for each ticket a player chooses six numbers out of 49). So, we'll start by building a function that calculates the probability of winning the big prize for any given ticket.

We discussed with the engineering team of the medical institute, and they told us we need to be aware of the following details when we write the function:

* Inside the app, the user inputs six different numbers from 1 to 49.
* Under the hood, the six numbers will come as a Python list, which will serve as the single input to our function.
* The engineering team wants the function to print the probability value in a friendly way — in a way that people without any probability training are able to understand.


```python
def one_ticket_probability():
    # nso:  number of successful outcomes
    # tnpo: total number of possible outcomes
    nso  = 1
    tnpo = combinations(49, 6)
    prob = (nso / tnpo * 100)
    print("Probability of Winning the Lottery with One Ticket: %d in %d (%0.9f)" % (nso, tnpo, prob))

#Example:
one_ticket_probability()
```

    Probability of Winning the Lottery with One Ticket: 1 in 13983816 (0.000007151)


# Historical Data Check for Canada Lottery

We'll focus on exploring the historical data coming from the Canada 6/49 lottery. The data set can be downloaded from Kaggle and it has the following structure:


```python
import pandas as pd

lottery_data = pd.read_csv("649.csv")
rows = lottery_data.shape[0]
cols = lottery_data.shape[1]

print("Number of Rows   : %4d \t" % rows)
print("Number of Columns: %4d \t" % cols)
print("List of column names: ", list(lottery_data.columns))

print("First 3 Rows:")
display(lottery_data.head(3))
print("Last 3 Rows:")
display(lottery_data.tail(3))
```

    Number of Rows   : 3665 	
    Number of Columns:   11 	
    List of column names:  ['PRODUCT', 'DRAW NUMBER', 'SEQUENCE NUMBER', 'DRAW DATE', 'NUMBER DRAWN 1', 'NUMBER DRAWN 2', 'NUMBER DRAWN 3', 'NUMBER DRAWN 4', 'NUMBER DRAWN 5', 'NUMBER DRAWN 6', 'BONUS NUMBER']
    First 3 Rows:



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRODUCT</th>
      <th>DRAW NUMBER</th>
      <th>SEQUENCE NUMBER</th>
      <th>DRAW DATE</th>
      <th>NUMBER DRAWN 1</th>
      <th>NUMBER DRAWN 2</th>
      <th>NUMBER DRAWN 3</th>
      <th>NUMBER DRAWN 4</th>
      <th>NUMBER DRAWN 5</th>
      <th>NUMBER DRAWN 6</th>
      <th>BONUS NUMBER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>649</td>
      <td>1</td>
      <td>0</td>
      <td>6/12/1982</td>
      <td>3</td>
      <td>11</td>
      <td>12</td>
      <td>14</td>
      <td>41</td>
      <td>43</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>649</td>
      <td>2</td>
      <td>0</td>
      <td>6/19/1982</td>
      <td>8</td>
      <td>33</td>
      <td>36</td>
      <td>37</td>
      <td>39</td>
      <td>41</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>649</td>
      <td>3</td>
      <td>0</td>
      <td>6/26/1982</td>
      <td>1</td>
      <td>6</td>
      <td>23</td>
      <td>24</td>
      <td>27</td>
      <td>39</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>


    Last 3 Rows:



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRODUCT</th>
      <th>DRAW NUMBER</th>
      <th>SEQUENCE NUMBER</th>
      <th>DRAW DATE</th>
      <th>NUMBER DRAWN 1</th>
      <th>NUMBER DRAWN 2</th>
      <th>NUMBER DRAWN 3</th>
      <th>NUMBER DRAWN 4</th>
      <th>NUMBER DRAWN 5</th>
      <th>NUMBER DRAWN 6</th>
      <th>BONUS NUMBER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3662</th>
      <td>649</td>
      <td>3589</td>
      <td>0</td>
      <td>6/13/2018</td>
      <td>6</td>
      <td>22</td>
      <td>24</td>
      <td>31</td>
      <td>32</td>
      <td>34</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3663</th>
      <td>649</td>
      <td>3590</td>
      <td>0</td>
      <td>6/16/2018</td>
      <td>2</td>
      <td>15</td>
      <td>21</td>
      <td>31</td>
      <td>38</td>
      <td>49</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3664</th>
      <td>649</td>
      <td>3591</td>
      <td>0</td>
      <td>6/20/2018</td>
      <td>14</td>
      <td>24</td>
      <td>31</td>
      <td>35</td>
      <td>37</td>
      <td>48</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>


The data set contains historical data for 3,665 drawings (each row shows data for a single drawing), dating from 1982 to 2018. For each drawing, we can find the six numbers drawn in the following six columns:

* NUMBER DRAWN 1
* NUMBER DRAWN 2
* NUMBER DRAWN 3
* NUMBER DRAWN 4
* NUMBER DRAWN 5
* NUMBER DRAWN 6

# Function for Historical Data Check

Previously, we focused on opening and exploring the Canada lottery data set. Right now, we're going to write a function that will enable users to compare their ticket against the historical lottery data in Canada and determine whether they would have ever won by now.

The engineering team told us that we need to be aware of the following details:

* Inside the app, the user inputs six different numbers from 1 to 49.
* Under the hood, the six numbers will come as a Python list and serve as an input to our function.
* The engineering team wants us to write a function that prints:
 * the number of times the combination selected occurred in the Canada data set; and
 * the probability of winning the big prize in the next drawing with that combination.

We'll now start working on writing this function.


```python
def extract_numbers(row):
    return set([row['NUMBER DRAWN %d' % i] for i in range(1, 7)])

def check_historical_occurence(user_numbers, winning_numbers):
    user_numbers   = set(user_numbers)
    matching_rows  = winning_numbers == user_numbers
    num_occurences = sum(matching_rows)
    
    print("Number of Occurences: %d" % num_occurences)
    one_ticket_probability()

winning_numbers = lottery_data.apply(extract_numbers, axis = 1)
check_historical_occurence([3, 41, 11, 12, 14, 43], winning_numbers)
```

    Number of Occurences: 1
    Probability of Winning the Lottery with One Ticket: 1 in 13983816 (0.000007151)


# Multi-ticket Probability

Lottery addicts usually play more than one ticket on a single drawing, thinking that this might increase their chances of winning significantly. Our purpose is to help them better estimate their chances of winning — on this screen, we're going to write a function that will allow the users to calculate the chances of winning for any number of different tickets.

We've talked with the engineering team and they gave us the following information:

* The user will input the number of different tickets they want to play (without inputting the specific combinations they intend to play).
* Our function will see an integer between 1 and 13,983,816 (the maximum number of different tickets).
* The function should print information about the probability of winning the big prize depending on the number of different tickets played.


```python
def multi_ticket_probability(num_tickets):
    # nso:  number of successful outcomes
    # tnpo: total number of possible outcomes
    nso  = num_tickets
    tnpo = combinations(49, 6)
    prob = (nso / tnpo * 100)
    print("Probability of Winning the Lottery with %d Tickets: %d in %d (%0.9f" 
          % (nso, nso, tnpo, prob) +"%)")

#Example:
for num_tickets in [10, 50, 90]:
    multi_ticket_probability(num_tickets)
    
```

    Probability of Winning the Lottery with 10 Tickets: 10 in 13983816 (0.000071511%)
    Probability of Winning the Lottery with 50 Tickets: 50 in 13983816 (0.000357556%)
    Probability of Winning the Lottery with 90 Tickets: 90 in 13983816 (0.000643601%)


# Less Winnings  Probability

We're going to write one more function to allow the users to calculate probabilities for two, three, four, or five winning numbers.

For extra context, in most 6/49 lotteries there are smaller prizes if a player's ticket match two, three, four, or five of the six numbers drawn. As a consequence, the users might be interested in knowing the probability of having two, three, four, or five winning numbers.

These are the engineering details we'll need to be aware of:

* Inside the app, the user inputs:
 * six different numbers from 1 to 49; and
 * an integer between 2 and 5 that represents the number of winning numbers expected

Our function prints information about the probability of having the inputted number of winning numbers.


```python
def probability_less_6(num_matches):
    # nso:  number of successful outcomes
    # tnpo: total number of possible outcomes
    a = combinations(6, num_matches)
    b = combinations(49 - num_matches, 6 - num_matches)
    nso  = a * b
    tnpo = combinations(49, 6)
    prob = (nso / tnpo * 100)
    print("Probability of Winning the Lottery with %d matches: %d in %d (%0.9f" 
          % (num_matches, nso, tnpo, prob) +"%)")
    
#Example:
probability_less_6(2)
```

    Probability of Winning the Lottery with 2 matches: 2675475 in 13983816 (19.132653061%)

