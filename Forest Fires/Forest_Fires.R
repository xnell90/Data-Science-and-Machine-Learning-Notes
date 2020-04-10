library(readr)
library(dplyr)
library(ggplot2)
library(purrr)

Forest_Fires_Data <- read_csv("Documents/Programming/Data Science/Forest_Fires/forestfires.csv")
View(Forest_Fires_Data)
"
X:     X-axis spatial coordinate within the Montesinho park map: 1 to 9
Y:     Y-axis spatial coordinate within the Montesinho park map: 2 to 9
month: Month of the year:                                        'jan' to 'dec'
day:   Day of the week:                                          'mon' to 'sun'
FFMC:  Fine Fuel Moisture Code index from the FWI system:        18.7 to 96.20
DMC:   Duff Moisture Code index from the FWI system:             1.1 to 291.3
DC:    Drought Code index from the FWI system:                   7.9 to 860.6
ISI:   Initial Spread Index from the FWI system:                 0.0 to 56.10
temp:  Temperature in Celsius degrees:                           2.2 to 33.30
RH:    Relative humidity in percentage:                          15.0 to 100
wind:  Wind speed in km/h:                                       0.40 to 9.40
rain:  Outside rain in mm/m2:                                    0.0 to 6.4
area:  The burned area of the forest (in ha):                    0.00 to 1090.84
"

ordered_months <- c('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
ordered_days   <- c('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')

Forest_Fires_Data <- Forest_Fires_Data %>%
  mutate(month = factor(month, levels = ordered_months)) %>%
  mutate(day = factor(day, levels = ordered_days))

Forest_Fires_Data_Day_of_Week <- Forest_Fires_Data %>% 
  group_by(day) %>%
  summarise(num_fires = n())

Forest_Fires_Data_Month <- Forest_Fires_Data %>%
  group_by(month) %>%
  summarise(num_fires = n())

ggplot(data = Forest_Fires_Data_Day_of_Week, aes(x = day, y = num_fires)) + 
  geom_bar(stat = 'identity') +
  labs(x = "Day",
       y = "Number of Fires",
       title = "Which days have the most forest fires?")

ggplot(data = Forest_Fires_Data_Month, aes(x = month, y = num_fires)) +
  geom_bar(stat = 'identity') + 
  labs(x = "Month",
       y = "Number of Fires",
       title = "Which months have the most forest fires?")

create_box_plot <- function(y) {
  ggplot(data = Forest_Fires_Data, aes_string(x = 'day', y = y)) +
    geom_boxplot() +
    labs(x = "Day", y = y)
}

y_var <- names(Forest_Fires_Data)[5:13]
map(y_var, create_box_plot)

create_scatter_plot <- function(x) {
  ggplot(data = Forest_Fires_Data, aes_string(x = x, y  = 'area')) +
    geom_point() +
    labs(x = x, y = 'area')
}

map(y_var[1:8], create_scatter_plot)