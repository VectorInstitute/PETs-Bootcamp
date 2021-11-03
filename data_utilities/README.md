## ConnexOntario Data Dictionary

### Main Contacts:  
Base data about contacts.  Fields are:

- CLIENT_PROFILE_ID:  identifier
- CONTACT_DAY (YYYYMMDD format)
- HOUR (24-hour clock)
- MINUTE
- MUNICIPALITY (Not Identified means no municipality information was collected)
- CONTACT_METHOD (values are:  E-Mail, FAX, In Person, Mail, Phone, Web Chat)
- IS_SUBSTANCE:  1 indicates this contact was related to substance abuse, 0 indicates no
- IS_MENTAL_HEALTH:  1 indicates this contact was related to mental health, 0 indicates no
- IS_PROBLEM_GAMBLING:  1 indicates this contact was related to problem gambling, 0 indicates no
- IS_OTHER:  1 indicates this contact was related to another issue besides substance abuse, mental health and/or problem gambling, 0 indicates no
- LENGTH_OF_CONTACT:  grouping (values are Under 4 minutes, 4 – 15 minutes, 16 – 30 minutes, Over 30 minutes)
- CONTACTOR_TYPE (values are:  Family, Friend, Other, Professional, Self, Student)
- CRISIS_IND:  Y indicates that the contact identified as being in crisis
- AGE
- GENDER (values are:  Female, Male, Non-Binary / Gender Fluid, Not Identified) – the age and gender fields describe the person for whom services are being sought

### Capacities:  

Information about the capacities of service types listed in our database.  Fields are:

-  FUNCTIONAL_CENTRE:  the service type
-  MALE: total capacity for male clients
-  FEMALE:  total capacity for female clients
-  UNDIFFERENTIATED:  total capacity for services that do not differentiate beds/service slots by gender

### Referrals:  

Information provided to contacts by service type

-  CLIENT_PROFILE_ID:  identifier
-  FUNCTIONAL_CENTRE:  the service type
-  REFERRALS:  total number of referrals for this service type

### Supplementary Resources: 
Referrals to related resources

-  CLIENT_PROFILE_ID:  identifier
-  RESOURCE_TYPE:  the type of supplementary resource
-  REFERRALS:  total number of referrals for this resource type

### Languages:
Languages identified by contacts (other than English)

-  CLIENT_PROFILE_ID:  identifier
-  LANGUAGE

### Ethnicity:
Ethnic groups identified by contacts

-  CLIENT_PROFILE_ID:  identifier
-  TARGET_POPULATION:  the ethnicity

### Target Groups: 
Target groups identified for the contact

-  CLIENT_PROFILE_ID:  identifier
-  TARGET_POPULAITON:  the target group

### Substances:
The substances identified by the contact

-  CLIENT_PROFILE_ID:  identifier
-  SUBSTANCE_GROUP:  the classification group for the substance identified (based on DSM-V)
-  SUBSTANCE:  the substance identified

### Diagnosis:
The diagnostic categories and diagnosis identified by the contact

-  CLIENT_PROFILE_ID:  identifier
-  DIAGNOSTIC_GROUP:  the classification group for the diagnosis identified (based on DSM-V)
-  DIAGNOSIS:  the diagnosis identified (if provided)

### Special Provisions:  
Any special needs identified for the contact

-  CLIENT_PROFILE_ID:  identifier
-  SPECIAL_PROVISION:  the special need identified
