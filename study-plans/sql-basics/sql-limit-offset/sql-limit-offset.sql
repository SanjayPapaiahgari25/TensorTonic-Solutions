-- Write your SQL query here
SELECT product, revenue, sale_date
FROM sales
ORDER BY revenue desc, sale_date
LIMIT 3
OFFSET 1