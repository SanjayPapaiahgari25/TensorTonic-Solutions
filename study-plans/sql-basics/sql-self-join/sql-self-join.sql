-- Returns: username, referrer_name.
SELECT A.USERNAME as username, COALESCE(B.USERNAME, 'organic') as referrer_name
FROM USER_REFERRALS A
LEFT JOIN USER_REFERRALS B
ON A.REFERRED_BY=B.ID
ORDER BY username