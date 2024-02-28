import logging
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Union


class CustomerSegmentationDataLoader(BaseModel):
    features: Union[Dict[str, Union[float, str]], List[Dict[str, Union[float, str]]]]
