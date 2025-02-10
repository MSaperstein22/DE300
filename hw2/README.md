Instructions for running etl_script.p:

1. Navigate to the hw2 folder with all necessary files.
2. Once Docker is running, run the following command: docker build -t etl-heart-disease .
3. Run the following command: docker run -it --rm --name etl-container etl-heart-disease
4. Read through the logs to see helpful print statements on the output of the binary classification models. For more information the ETL process, un-comment out the print statements in etl_script.py.