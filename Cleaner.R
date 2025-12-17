# =======================
# 3.2 DATA PREPROCESSING
# =======================

# ---- Data Import and Initial Cleaning ----
library(readr)
library(forecast)
library(tseries)

data <- read_csv("./sales_data.csv")

data$Daily_Sales_Revenue <- as.numeric(data$Daily_Sales_Revenue)
data$Daily_Sales_Revenue <- na.interp(data$Daily_Sales_Revenue)


# ---- Outlier Treatment (IQR-based soft bounding) ----
Q1 <- quantile(data$Daily_Sales_Revenue, 0.25)
Q3 <- quantile(data$Daily_Sales_Revenue, 0.75)
IQR_val <- IQR(data$Daily_Sales_Revenue)

lower <- Q1 - 1.5 * IQR_val
upper <- Q3 + 1.5 * IQR_val

data$Daily_Sales_Revenue <- pmin(
  pmax(data$Daily_Sales_Revenue, lower),
  upper
)


# ---- Variance Stabilization (Log Transformation) ----
data$Daily_Sales_Revenue <- log(data$Daily_Sales_Revenue + 1)


# ---- Time Series Construction ----
ts_data <- ts(
  data$Daily_Sales_Revenue,
  frequency = 1
)


# ---- Stationarity Assessment ----
adf.test(ts_data)


# ---- Differencing for Trend Removal ----
ts_diff <- diff(ts_data, differences = 1)


# ---- Final Series for Modeling ----
ts_final <- ts_diff

# ---- Export Cleaned Dataset ----
clean_data <- data.frame(
  Date = data$Date,  # remove if not present
  Daily_Sales_Revenue = data$Daily_Sales_Revenue,
  Marketing_Spend = data$Marketing_Spend
)

write_csv(clean_data, "./cleaned_sales_data.csv")

