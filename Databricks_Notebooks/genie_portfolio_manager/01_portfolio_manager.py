# Databricks notebook source
# MAGIC %md
# MAGIC ## Data access
# MAGIC
# MAGIC For the purpose of that demo, we will be using 2 sample datasets from our databricks marketplace, news and daily stopck prices. Although those delta shares might not be publicly available, these can be easily swapped with others. Please check our listings [here](https://marketplace.databricks.com/?searchKey=prices&sortBy=date).
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/market_data.png' width=500>
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/market_news.png' width=500>

# COMMAND ----------

# MAGIC %md
# MAGIC Available as delta sharing, tables can be shared to a dedicated catalog (respectively `fsgtm_market_news` and `fsgtm_market_data` here). 

# COMMAND ----------

# Reading from 2 public shares on marketplace
db_catalog_news = 'fsgtm_market_news'
db_catalog_market = 'fsgtm_market_data'

# COMMAND ----------

display(sql(f'SHOW DATABASES IN {db_catalog_news}'))

# COMMAND ----------

display(sql(f'SHOW TABLES IN {db_catalog_news}.market_data'))

# COMMAND ----------

display(sql(f'DESCRIBE TABLE {db_catalog_news}.market_data.news'))

# COMMAND ----------

# MAGIC %md
# MAGIC Unfortunately, these tables are poorly described, making text-to-sql capability more complex. In the next section, we will materialize those tables with properly defined metadata and column descriptions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create data model
# MAGIC Data shared by provider might not contain description and metadata that can be further leveraged by Databricks AI capabilities. In this section, we will enforce schema and metadata by physically materializing those delta shares into tables, all governed on unity catalog (`fsgtm`). Please change catalog name as your convenience.

# COMMAND ----------

# Writing to a dedicated catalog
genie_catalog = 'fsgtm'
genie_schema = 'genie_cap_markets'

# COMMAND ----------

_ = sql(f'CREATE CATALOG IF NOT EXISTS {genie_catalog}')
_ = sql(f'CREATE DATABASE IF NOT EXISTS {genie_catalog}.{genie_schema}')

# COMMAND ----------

# MAGIC %md
# MAGIC By creating tables, it is worth mentioning that Databricks can automatically suggest comlumn and table description using generative AI. Simply open your table from your catalog (in this case `fsgtm.genie_cap_market.news`) as follows. For the purpose of our demo, we will provide column description at table creation, ensuring relevance of our metadata for text-to-sql capabilities.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/metadata_generate.png' width=800/>

# COMMAND ----------

_ = sql(f'''CREATE OR REPLACE TABLE {genie_catalog}.{genie_schema}.portfolio (   
  `ticker`              STRING COMMENT 'Unique identifier for the stock, allowing easy reference and tracking.',
  `company_name`        STRING COMMENT 'The name of the company, providing a recognizable and human-readable label for identification.',
  `company_description` STRING COMMENT 'A brief overview of the company, its products, and services, generated using DBRX model.',
  `company_website`     STRING COMMENT 'The official website of the company, providing more detailed information and resources.',
  `company_logo`        STRING COMMENT 'The visual representation of the company, allowing for easy recognition and branding.',
  `industry`            STRING COMMENT 'The industry or sector in which the company operates, providing context for the company business and potential competitors.
'
) USING DELTA
COMMENT 'The `portfolio` table contains information about the companies in our investment portfolio. It includes details about the company ticker, name, description, website, logo, and industry. This data can be used to analyze the portfolio composition, monitor industry trends, and perform research on individual companies. It can also be used to generate reports and visualizations for stakeholders, such as the portfolio diversity and the performance of companies in specific industries.'
''')

# COMMAND ----------

_ = sql(f'''CREATE OR REPLACE TABLE {genie_catalog}.{genie_schema}.fundamentals (   
  `ticker`                STRING  COMMENT 'Unique identifier for the stock, allowing easy reference and tracking.',
  `market_capitalization` DOUBLE  COMMENT 'Represents the current market capitalization of the stock, indicating the stock value in the market.',
  `outstanding_shares`    DOUBLE  COMMENT 'Represents the number of outstanding shares of the stock, indicating the liquidity and ownership of the stock.'
) USING DELTA
COMMENT 'The `fundamentals` table contains fundamental information about various stocks, including market capitalization and outstanding shares. This data can be used to analyze the financial health of individual stocks, as well as to compare the performance of different stocks over time. It can also be used to identify trends in market capitalization and outstanding shares, which can inform investment decisions and market analysis.'
''')

# COMMAND ----------

_ = sql(f'''CREATE OR REPLACE TABLE {genie_catalog}.{genie_schema}.prices (   
  `ticker`          STRING COMMENT 'Unique identifier for the stock or security, allowing easy reference and tracking.',
  `date`            DATE COMMENT 'The date for which the price information is provided.',
  `open`            DOUBLE COMMENT '
Represents the opening price of the security on the given date.',
  `high`            DOUBLE COMMENT 'Represents the highest price of the security on the given date.',
  `low`             DOUBLE COMMENT 'Represents the lowest price of the security on the given date.',
  `close`           DOUBLE COMMENT 'Represents the closing price of the security on the given date.',
  `adjusted_close`  DOUBLE COMMENT 'Represents the adjusted closing price of the security on the given date, accounting for any corporate actions or other adjustments. This represents the cash value of the last transacted price before the market closes',
  `return`          DOUBLE COMMENT 'Represents the return of the security on the given date, calculated as the difference between the closing price and last day closing price.',
  `volume`          DOUBLE COMMENT 'Represents the trading volume of the security on the given date, indicating the number of shares traded.',
  `split_factor`    DOUBLE COMMENT 'Represent the stock split of a given ticker at any point in time'
) USING DELTA
COMMENT 'The `prices` table contains stock price data for various tickers. It includes information on daily open, high, low, and closing prices, as well as adjusted closing prices, returns, and trading volumes. This data can be used for stock analysis, trend identification, and risk assessment. It can also be used to generate reports and visualizations for stakeholders to monitor market trends and make informed decisions.'
''')

# COMMAND ----------

_ = sql(f'''CREATE OR REPLACE TABLE {genie_catalog}.{genie_schema}.news_ticker (   
  `ticker`     STRING COMMENT 'Unique identifier for the stock ticker, allowing easy reference and tracking of specific stocks.',
  `article_id` STRING COMMENT 'Identifier for the news article related to the stock ticker, enabling linking articles to their respective tickers.'
) USING DELTA
COMMENT 'The `news_ticker` table contains information about ticker symbols and the corresponding news articles. It can be used to track news related to various ticker symbols, enabling users to monitor market trends and news that may impact the performance of their investments. This table can also be used to identify ticker symbols associated with specific news articles, making it easier to analyze the impact of news on financial markets.'
''')

# COMMAND ----------

_ = sql(f'''CREATE OR REPLACE TABLE {genie_catalog}.{genie_schema}.news (   
  `article_id`        STRING     COMMENT 'Unique identifier for each news article.',
  `published_time`    TIMESTAMP  COMMENT 'The time when the article was published.',
  `source`            STRING     COMMENT 'The news source or publisher that published the article.',
  `source_url`        STRING     COMMENT 'The URL of the article, allowing users to access the original content.',
  `title`             STRING     COMMENT 'The title of the news article, providing a brief overview of the content.',
  `sentiment`         DOUBLE     COMMENT 'Represents the sentiment or tone of the article, measured as a double value between -1 (negative) and 1 (positive).',
  `market_sentiment`  STRING     COMMENT 'Represents the market sentiment for a given article, can be Bearish, Bullish, or Neutral'
) USING DELTA
COMMENT 'The `news` table contains articles from various sources related to the financial markets. It includes details such as the article title, the source, and the sentiment of the article. This data can be used to monitor market trends, track sentiment changes, and analyze the impact of different news sources on market behavior. This information can be particularly useful for traders and analysts who need to stay up-to-date with market news and understand how it might affect their investments.'
''')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest data
# MAGIC With our tables properly defined and governed on UC, we can materialize our delta shares into tables as simple select statement. We will ingest the following tables
# MAGIC
# MAGIC - portfolio: contains companies description for an hypothetical portfolio
# MAGIC - fundamental: contains market fundamentals such as market cap for each ticker
# MAGIC - prices: daily prices for each ticker, including open, high, low, close and volume information
# MAGIC - news: news articles mentioning our different instruments
# MAGIC - news_ticker: article may cover multiple tickers

# COMMAND ----------

insert = sql(f'''INSERT INTO {genie_catalog}.{genie_schema}.fundamentals
SELECT 
  ticker,
  marketCap AS market_capitalization,
  sharesOutstanding AS outstanding_shares
FROM 
  {db_catalog_market}.market_data.company_profile
WHERE 
  ticker IS NOT NULL
  AND ticker != 'NaN'
  AND companyName IS NOT NULL''')

display(insert)

# COMMAND ----------

# MAGIC %md
# MAGIC An interesting feature to explore is our AI functions (see [doc](https://docs.databricks.com/en/large-language-models/ai-functions.html)). Our portfolio dataset does not contain company description but company name, country and industry. Given the foundational knowledge of modern LLM such as DBRX, we can easily delegate the task of generating a company description to a large language model using `ai_query()` function

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC ai_query(
# MAGIC     'databricks-dbrx-instruct',
# MAGIC     concat_ws(' ', 'Describe company Apple Inc. in the technology industry')
# MAGIC   ) AS company_description

# COMMAND ----------

insert = sql(f'''INSERT INTO {genie_catalog}.{genie_schema}.portfolio
SELECT 
  ticker,
  companyName AS company_name,
  ai_query(
    'databricks-dbrx-instruct',
    concat_ws(' ', 'Describe company', companyName, 'in the', industry, 'industry')
  ) AS company_description,
  website AS company_website,
  logo AS company_logo,
  industry
FROM 
  {db_catalog_market}.market_data.company_profile
WHERE 
  ticker IS NOT NULL
  AND ticker != 'NaN'
  AND companyName IS NOT NULL''')

display(insert)

# COMMAND ----------

display(spark.read.table(f'{genie_catalog}.{genie_schema}.portfolio'))

# COMMAND ----------

# MAGIC %md
# MAGIC In order to ensure relevance of this data for portfolio manager, we simply compute instrument return as a simple window function
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```
# MAGIC return = (close_price / last_close_price) - 1
# MAGIC ```

# COMMAND ----------

insert = sql(f'''INSERT INTO {genie_catalog}.{genie_schema}.prices
SELECT 
  ticker,
  `date`,
  `open`,
  `high`,
  `low`,
  `close`,
  `adjClose` AS `adjusted_close`,
  -- compute investment return as a window function
  adjClose/(lag(adjClose) OVER (PARTITION BY ticker ORDER BY `date`))-1 AS `return`,
  `vol` AS `volume`,
  `splitFactor` AS `split_factor`
FROM 
  {db_catalog_market}.market_data.dailyprice
WHERE 
  ticker IS NOT NULL''')

display(insert)

# COMMAND ----------

insert = sql(f'''INSERT INTO {genie_catalog}.{genie_schema}.prices
SELECT 
  ticker,
  `date`,
  `open`,
  `high`,
  `low`,
  `close`,
  `adjClose` AS `adjusted_close`,
  `vol` AS `volume`,
  `splitFactor` AS `split`
FROM 
  {db_catalog_market}.market_data.dailyprice
WHERE 
  ticker IS NOT NULL''')

display(insert)

# COMMAND ----------

insert = sql(f'''INSERT INTO {genie_catalog}.{genie_schema}.news_ticker
SELECT 
  ticker,
  articleId AS article_id 
FROM 
  {db_catalog_news}.market_data.news_ticker_sentiment''')

display(insert)

# COMMAND ----------

insert = sql(f'''INSERT INTO {genie_catalog}.{genie_schema}.news
SELECT DISTINCT
  articleId AS article_id,
  publishedTime AS published_time,
  source,
  `url` AS source_url,
  title AS title,
  articleSentimentScore AS sentiment,
  articleSentimentLabel AS market_sentiment  
FROM 
  {db_catalog_news}.market_data.news''')

display(insert)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Room
# MAGIC With only a few steps, our set of tables are ready for genie data room. Head over to genie, and select different tables from your catalog.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/genie_create.png' width=800/>

# COMMAND ----------

# MAGIC %md
# MAGIC At this point, you can ask genie for specific questions, such as:

# COMMAND ----------

# MAGIC %md
# MAGIC **What are the top 5 companies by market capitalization in our investment portfolio?**
# MAGIC This query calculates the total market capitalization for each industry represented in the portfolio. It joins the `portfolio` table with the `fundamentals` table on the `ticker` column to get the market capitalization for each company. Then, it groups the results by `industry` and sums up the `market_capitalization` for each industry to get the total market cap per industry. One can easily visualize as a pie chart by asking Genie.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/portfolio_diversification.png' width=800/>

# COMMAND ----------

# MAGIC %md
# MAGIC **How has the stock price of Apple changed over time?**
# MAGIC This query should understand that Apple correspond to ticker `AAPL` and retrieve information from `prices` table. One can notice a major drop in 2014-2015. So why not asking genie if **there was a stock split for AAPL in the 2014 2015 timeframe?**
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/aapl_price.png' width=800/>

# COMMAND ----------

# MAGIC %md
# MAGIC There might be some concepts that a model may not be completely familiar with. Asking a question about market volatility may yield incorrect results. Luckily, one can create additional set of instructions that can guide our model towards the right answer. In this case, we explicitly mention volatily to be a measure of standard deviation. These instructions will be borught in as additional context for our LLM
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/genie_instruction.png' width=800/>

# COMMAND ----------

# MAGIC %md
# MAGIC **Show me the market volatility for Technology companies during the financial crisis by week**
# MAGIC Modern LLMs should be smart enough to understand the concept of "financial crisis", translating this question as the following query, leveraging our definition of market volality as per our genie instruction
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC
# MAGIC ```sql
# MAGIC SELECT
# MAGIC   date_sub (date, DAYOFWEEK (date) - 1) AS week_start,
# MAGIC   STDDEV (`return`) AS volatility
# MAGIC FROM
# MAGIC   fsgtm.genie_cap_markets.prices p
# MAGIC   JOIN fsgtm.genie_cap_markets.portfolio po ON p.ticker = po.ticker
# MAGIC WHERE
# MAGIC   po.industry = 'Technology'
# MAGIC   AND date BETWEEN '2007-12-01' AND '2009-06-30'
# MAGIC GROUP BY
# MAGIC   week_start
# MAGIC ORDER BY
# MAGIC   week_start
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC **What was the most traded stock during covid?** Same as above, model may be incredibly smart to understand the context of the question. This query calculates the total trading volume for each stock during the COVID-19 period defined as March 1, 2020, to December 31, 2020. It groups the data by the ticker symbol, sums up the trading volume for each group, and then orders the results in descending order based on the total trading volume. The query then limits the results to only the top record, which represents the most traded stock during the specified period.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example questions
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/genie_portfolio_manager/main/images/genie_demo.gif'/>

# COMMAND ----------

# MAGIC %md
# MAGIC - *How diversified is my portfolio by market cap?*
# MAGIC - *Visualize as a pie chart*
# MAGIC - *What are the top 5 companies by market capitalization in our investment portfolio?*
# MAGIC - *Visualize*
# MAGIC - *How has the stock price of Apple changed over time?*
# MAGIC - *Visualize*
# MAGIC - *Was there a stock split for AAPL in the 2014 2015 timeframe?*
# MAGIC - *Show me the market volatility for Technology companies during the financial crisis by week*
# MAGIC - *Visualize*
# MAGIC - *What is the market sentiment for companies in the retail industry?*
# MAGIC - *Visualize as a bar chart*
# MAGIC - *What was the top 10 most recent negative event for banking companies in my portfolio*
# MAGIC - *How has the market sentiment changed by week for my portfolio?*
# MAGIC - *Visualize*
# MAGIC - *Tanks Genie, goodbye!*

# COMMAND ----------

# MAGIC %md
# MAGIC
