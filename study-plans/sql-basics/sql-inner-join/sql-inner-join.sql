-- Write your SQL query here
SELECT name, salary, dept_name
FROM EMPLOYEES A
INNER JOIN DEPARTMENTS B
ON A.dept_id = B.id
ORDER BY name