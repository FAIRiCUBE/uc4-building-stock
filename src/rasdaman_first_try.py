#!/usr/bin/env python
# coding: utf-8

# # Rasdaman Data Access - First Tries
# 
# *Rob Knapen, Wageningen Environmental Research*
# 
# This notebook covers some first tries with using Rasdaman coverage services (WCS and WCPS).

# In[4]:


# this notebook uses dotenv to retrieve secrets from environment variables
# get it with: pip install python-dotenv

# get_ipython().run_line_magic('reload_ext', 'dotenv')
# get_ipython().run_line_magic('dotenv', '')


# In[5]:

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio.plot import show_hist
import requests
import json
import xmltodict


plt.close('all')

load_dotenv() 

# read the secret information ...
#   Create a .env file in the notebooks folder first that has entries like:
#
#   RASDAMAN_SERVICE_ENDPOINT=<url>
#   RASDAMAN_CRED_USERNAME=<rasdaman_endpoint_username>
#   RASDAMAN_CRED_PASSWORD=<rasdaman_endpoint_password>
#   etc.

rasdaman_username = os.getenv("RASDAMAN_CRED_USERNAME")
rasdaman_password = os.getenv("RASDAMAN_CRED_PASSWORD")
rasdaman_endpoint = os.getenv("RASDAMAN_SERVICE_ENDPOINT")

base_wcs_url = rasdaman_endpoint + "?service=WCS&version=2.1.0"


# ## Using Web Coverage Service (WCS)

# ### WCS - GetCapabilities

# In[7]:




# retrieve a description of the WCS using the GetCapabilities request
response = requests.get(base_wcs_url + "&request=GetCapabilities", auth=(rasdaman_username, rasdaman_password))
wcs_capabilities = xmltodict.parse(response.content)
# print(json.dumps(wcs_capabilities, indent=2))


# In[8]:


# list the WCS contents
wcs_coverage_summary = wcs_capabilities['wcs:Capabilities']['wcs:Contents']['wcs:CoverageSummary']
print(json.dumps(wcs_coverage_summary, indent=2))


# ### WCS - DescribeCoverage

# In[9]:


# retrieve a description of a coverage using the DescribeCoverage request
coverage_id = wcs_coverage_summary['wcs:CoverageId']

response = requests.get(base_wcs_url + "&request=DescribeCoverage&coverageId=" + coverage_id,
                        auth=(rasdaman_username, rasdaman_password)
                        )
wcs_coverage_description = xmltodict.parse(response.content)
# print(json.dumps(wcs_coverage_description, indent=2))


# ### WCS - GetCoverage

# In[10]:


# Sandefjord region
# subset_lat = "&subset=Lat(59.10, 59.20)"
# subset_long = "&subset=Long(10.20, 10.30)"

# Tromso region
subset_lat = "&subset=Lat(69.60, 69.70)"
subset_long = "&subset=Long(18.90, 19.00)"

cov_id = "&COVERAGEID=" + coverage_id
encode_format = "&FORMAT=tiff"

response = requests.get(
    base_wcs_url + "&request=GetCoverage" + cov_id + subset_long + subset_lat + encode_format,
    auth=(rasdaman_username, rasdaman_password),
    verify=False)

wcs_coverage_data = response.content

# show content if there was an error
if not response.ok:
    dict = xmltodict.parse(wcs_coverage_data)
    print(json.dumps(dict, indent=2))


# In[11]:


with MemoryFile(wcs_coverage_data) as memfile:
    with memfile.open() as ds:
        print(ds.name, ds.count, ds.width, ds.height)
        print(ds.bounds)
        ds_array = ds.read(1)
        # show_hist(ds_array, bins=50, lw=0.0, stacked=False, alpha=0.3, histtype='stepfilled')
        plt.imshow(ds_array, cmap='coolwarm_r')


# In[12]:


# another way to visualise the retrieved coverage data
# from IPython.display import Image
# Image(data=wcs_coverage_data)


# ## Using Web Coverage Processing Service (WCPS)

# In[13]:


query = '''
for $c in (imperviousness)
return
  encode(
    $c[ Lat(69.60, 69.70), Long(18.90, 19.00)" ],
    "image/tiff"

  )
'''

response = requests.get(
    base_wcs_url + "&request=ProcessCoverage&query=" + query,
    auth=(rasdaman_username, rasdaman_password),
    verify=False)

wcps_coverage_data = response.content

# show content if there was an error
if not response.ok:
    dict = xmltodict.parse(wcps_coverage_data)
    print(json.dumps(dict, indent=2))


# In[ ]:




