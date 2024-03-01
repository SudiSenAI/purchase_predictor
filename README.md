# purchase_predictor
StateFarm's GLM model for predicting if the customer is likely to purchase an item


Linear Regression Model for predicting customer purchasing tendencies based on 100 masked features
Features are normalized using StandardScaler and imputed with mean values.


The model code is wrapped inside an API: The model is made callable via API call (port 8000).
The call will pass 1 to N rows of data in JSON format, and expects a N responses each with a predicted class and probability belonging to the predicted class. 
For Example:

curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}'

or a batch curl call:

curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '[{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"},{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}]'

Each of the 10,000 rows in the test dataset will be passed through an API call. 
The call could be a single batch call w/ all 10,000 rows, or 10,000 individual calls.
API should be able to handle either case with minimal impact to performance.

### Deployment:
Using Kubernetes load balancing of API wrapped docker containers

