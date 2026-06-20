-- Write your SQL query here
SELECT name, city, COALESCE(SUM(amount), 0) AS total_spent
FROM customers C
LEFT JOIN orders O
ON C.id=O.customer_id
GROUP BY name, city
ORDER BY total_spent DESC,name