FAIRiCUBE Use Case "Stock modeling of buildings"
==============================

## About this use case
Buildings are responsible for about 40% and 36% of energy demand and greenhouse gas (GHG) emissions, respectively, in the EU. Reducing the energy demand and environmental impacts associated with buildings have a crucial role in achieving the EU’s energy and climate goals. “Renovation wave strategy” and “fit for 55” are a set of policies aiming to pave the way to make the European building stocks energy-efficient and less carbon-intensive. While enhancing energy efficiency and having environmentally sound buildings assist the EU’s climate neutrality goals, the need for a circular use of building materials become more prominent. Huge amounts of materials are consumed and stocked during the entire lifetimes of buildings, which makes buildings a mining asset for future supply of materials. Adopting a circular economy principle may assist in both reducing environmental impacts associated with the production of building materials and increasing the resilience to supply chain disruption as well as avoiding increased prices of raw materials.

Ensuring that the EU’s climate and energy objectives are achievable and the inventory of in-use building materials are correctly mapped, requires a sufficient knowledge of the properties of our built environment. However, what one can often find is a mix of detailed and generic data. For newly built buildings, due to current intensive regulative codes (particularly in the EU), a prominent level of clarity is demanded giving the possibility to understand the properties of assets. However, for older buildings, a detailed level of clarity is often scarce until assets are (deeply) refurbished, rehabilitated, or inspected. This mixed level of insights on the properties of buildings constrains the possibility of making decisions at the national level to prioritize dwellings with the highest return on investments and promotion of circular and local materials.

Within this context the primary goal of UC4 is to gain a deeper knowledge on the energy and environmental performance of buildings, as well as in-use of natural resources. The gained knowledge will also reflect on three of Green Deal Action Plans which are connected to sustainability performance of buildings: (1) circular economy, (2) climate neutrality, and (3) renovation wave. The objectives of this use case are twofold. Firstly, developing and documenting models based on data compliant with the F.A.I.R-data definition to estimate (a) material use intensity and energy performance and (b) associated greenhouse gas emissions of building stocks. Secondly, mapping the building estimates on a datacube structure. 

## Research questions
The following research questions strive to reach UC4's objectives:

+ How can the inconsistency of data availability across European cities be enhanced by machine learning models? And what independent variables are rudimentary for training the models?
+ To what extent can data cube infrastructure support actors, researchers, stakeholders, etc. to tackle the Green Deal priority action plans related to climate change, renovation wave, and circular economy?
+ How efficiently and effectively can data cube infrastructure be scaled up to cover building stock at the regional or national level?
+ Can data cube infrastructure knowledge be easily transferred and reproduced for other types of infrastructure (i.e., green, blue, and grey infrastructure)?

## Data Analysis
The use case will output raster images containing energy and environmental performance estimates, as well as potential stocks of building materials. The estimations will be narrowed down to four European cities. Oslo, Barcelona, Luxembourg, and Vienna are considered to assure the diversity of the cities from the architectural and climate zone viewpoints.

Vector and raster data from publicly available platforms (e.g., OpenStreetMap and INSPIRE geoportal) will be used in the first place to create the 3D models. Such public data should contain the necessary information to create 3D models like building footprints, construction year, building height, etc. However, in case of missing the necessary data from the public platforms, local data from public bodies will be considered. Since the cities of choice in this use case have local representatives in this consortium, there is a higher likelihood of bridging the data gap. However, if contacting local bodies doesn’t yield a suitable outcome, the work will expand its scope by means of other data fulfilling the F.A.I.R-data definition, like satellite data. In addition to the identification and collection of data, suitable machine-learning techniques will be used for the gap-filling of missing information.




