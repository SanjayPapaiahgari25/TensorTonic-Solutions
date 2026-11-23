-- Returns: username, experiment_name, variant, revenue.
SELECT username,experiment_name,variant,revenue
FROM users U
INNER JOIN experiment_assignments E
ON U.id=E.user_id 
INNER JOIN conversions C
ON U.id=C.user_id
ORDER BY experiment_name,revenue DESC,username