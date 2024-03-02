from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.utils import get_collection_dataframe
import os
import sys

# def test_logger_exception():
#     try:
#         logging.info("Starting the test logger and exception!")
#         result = 3 / 0
#         print(result)
#         logging.info("Ending point of the test logger and exception!")
#     except Exception as e:
#         logging.debug(str(e))
#         raise InsuranceException(e, sys)
    

if __name__ == "__main__":
    try:
        # test_logger_exception()
        get_collection_dataframe(databaseName="INSURANCE", collectionName="INSURANCE_COLLECTION")

    except Exception as e:
        print(e)