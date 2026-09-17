-- Online Retail Customer Analytics: RFM segmentation (PostgreSQL)
-- Dibimbing Business Intelligence Bootcamp, Batch 14 capstone
-- Source: UCI Online Retail II (https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
-- Input: cleaned transactions (rows without Customer_ID and non-positive quantities removed),
--        with Revenue = Quantity * Price computed during pre-processing.
-- Output: every transaction tagged with its customer's RFM segment, exported to Tableau
--         for the executive dashboard.

-- Create table for e-commerce invoice data
CREATE TABLE ecommerce_invoices (
    Invoice INTEGER NOT NULL,      -- Invoice number (e.g., 48934)
    StockCode VARCHAR(20),         -- Product stock code (e.g., '85048', '79323P', '79323W')
    Description VARCHAR(255),      -- Product description
    Quantity INTEGER,              -- Quantity of items purchased
    InvoiceDate DATE NOT NULL,     -- Date of invoice (YYYY-MM-DD format)
    Price DECIMAL(10,2),           -- Unit price of the item
    Customer_ID VARCHAR(20),       -- Customer identifier (e.g., 13085.0)
    Country VARCHAR(50),           -- Customer country
    Revenue DECIMAL(12,2)          -- Quantity * Price
);

with

step_1 as ( -- aggregate RFM metrics at customer level
select
		customer_id,
		current_date - max(InvoiceDate) as recency,
		count(distinct stockcode) as freq,
		avg(revenue) as monetary
from	public.ecommerce_invoices ei
group by 1
)

,step_2 as ( -- score each metric into quartiles (1 = best)
select
		*,
		ntile(4) OVER (ORDER BY recency) AS q_recency,
		ntile(4) OVER (ORDER BY freq desc) AS q_freq,
		ntile(4) OVER (ORDER BY monetary desc) AS q_monetary
from	step_1
)

,step_3 as ( -- combine quartiles into an RFM score
select
		*
		,concat(q_recency,q_freq,q_monetary) as rfm_score
from	step_2
)

-- Best Customer (111)
-- Potential Customers (112, 122, 211, 222)
-- Lost Cheap (444, 443, 434)

-- Big Spender (XX1)
-- Loyal Customers (X1X)
-- Others/Recent Shopper (1XX & 2XX)
-- Almost Lost (3XX)
-- Lost Customers (4XX)

,step_final as ( -- assign RFM segment
select
		customer_id,
		rfm_score,
		case when rfm_score = '111' then 'Best Customer'
			when rfm_score in ('112','122','211','222') then 'Potential Customers'
			when rfm_score in ('444', '443', '434') then 'Lost Cheap'

			when q_monetary = 1 then 'Big Spender'
			when q_freq = 2 then 'Loyal Customers'
			when q_recency in (1,2) then 'Others/Recent Shopper'
			when q_recency = 3 then 'Almost Lost'
			when q_recency = 4 then 'Lost Customers'
			end as rfm_segment
from	step_3
)

select
			ei.*,
			sf.rfm_segment
from		public.ecommerce_invoices as ei
left join 	step_final as sf
			on ei.customer_id = sf.customer_id
