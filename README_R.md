# IPL Match Winner Prediction - R Shiny Application

This is an R Shiny implementation of the IPL Match Winner Prediction application, converted from the original Python Flask version.

## Features

- Predicts the winner of IPL 2025 matches based on various factors
- Uses Random Forest machine learning model for prediction
- Interactive web interface built with Shiny
- Shows prediction probability and explanation for the predicted outcome
- Includes visual representation of win probabilities

## Prerequisites

Make sure you have R installed on your system. You can download it from [CRAN](https://cran.r-project.org/).

## Required R Packages

Install the following packages using the command `install.packages(c("package_name"))`:

```R
install.packages(c("shiny", "shinythemes", "dplyr", "tidyr", "readr", "caret", "randomForest"))
```

## How to Run the Application

1. Open RStudio or your preferred R environment
2. Set your working directory to the folder containing `app.R`
3. Run the application using one of these methods:
   - In RStudio: Click the "Run App" button
   - In R console: Run `shiny::runApp("app.R")`

## Usage Instructions

1. Select the two competing teams
2. Choose the match venue
3. Select which team won the toss
4. Specify the toss decision (bat or field)
5. Click "Predict Winner" to get the prediction result

## Data

The application uses:
- Historical IPL match data from `ipl.csv` if available
- If not available, it will generate sample data for demonstration purposes

## Converting From Python to R

The application was converted from a Python Flask web application to an R Shiny application. Key components that were reimplemented:

1. **Data Handling**: Using tidyverse packages for data manipulation
2. **Machine Learning**: Replaced Python's LightGBM model with R's RandomForest
3. **User Interface**: Converted HTML/CSS to Shiny UI components
4. **Server Logic**: Reimplemented the prediction logic in R

## Credits

Original Python version by: [User]
R conversion by: [Your Name] 