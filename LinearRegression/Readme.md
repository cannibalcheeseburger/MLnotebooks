# Dataset Description
## House Features

 - `INT_SQFT` – The interior Sq. Ft of the property
 - `N_BEDROOM` – The number of Bed rooms
 - `N_BATHROOM` - The number of bathrooms
 - `N_ROOM` – Total Number of Rooms
 - `QS_ROOMS` – The quality score assigned for rooms based on buyer reviews
 - `QS_BATHROOM` – The quality score assigned for bathroom based on buyer reviews
 - `QS_BEDROOM` – The quality score assigned for bedroom based on buyer reviews
 - `QS_OVERALL` – The Overall quality score assigned for the property
 - `SALE_COND` – The Sale Condition
    - Normal: Normal Sale
    - Abnorml: Abnormal Sale - trade,  -       foreclosure, short sale
    - AdjLand: Adjoining Land Purchase
    - Family: Sale between family members
    - Partial: Home was not completed when last assessed
 - `BUILDTYPE` – The type of building
    - House
    - Commercial
    - Others

## Surrounding and Locality

 - `AREA` – The property in which the real estate is located
 - `DIST_MAINROAD` – The distance of the property to the main road
 - `PARK_FACIL` – Whether parking facility is available
 - `UTILITY_AVAIL`
    - AllPub`: All public Utilities (E,G,W,& S)
    - NoSewr`: Electricity, Gas, and Water (Septic Tank)
    - NoSeWa`: Electricity and Gas Only
    - ELO`: Electricity only
 - `STREET`
    - Gravel
    - Paved
    - No Access
 - `MZZONE`
    - A: Agriculture
    - C: Commercial
    - I: Industrial
    - RH: Residential High Density
    - RL: Residential Low Density
    - RM: Residential Medium Density

## House Sale Price

 - `PRT_ID` – The Property Transaction ID assigned by ChennaiEstate
 - `COMMIS` – The Commission paid to the agent
 - `ALES_PRICE` – The total sale price of the property