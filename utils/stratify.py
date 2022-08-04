import pandas as pd
import numpy as np

admimeth = {
"11": "Waiting list",
"12": "Booked",
"13": "Planned",
"21": "Accident and emergency or dental casualty department of the Health Care Provider",
"22": "GENERAL PRACTITIONER", #  after a request for immediate admission has been made direct to a Hospital Provider, i.e. not through a Bed bureau, by a GENERAL PRACTITIONER or deputy
"23": "Bed bureau",
"24": "Consultant Clinic, of this or another Health Care Provider",
"25": "Admission via Mental Health Crisis Resolution Team",
"2A": "Accident and Emergency Department of another provider where the PATIENT had not been admitted",
"2B": "Transfer of an admitted PATIENT from another Hospital Provider in an emergency",
"2C": "Baby born at home as intended",
"2D": "Other emergency admission",
"28": "Other means",
"31": "Admitted ante partum",
"32": "Admitted post partum",
"82": "The birth of a baby in this Health Care Provider",
"83": "Baby born outside the Health Care Provider except when born at home as intended",
"81": "Transfer of any admitted PATIENT from other Hospital Provider other than in an emergency"
}
admimeth = {'12': 'Booked',
            '24': 'Consultant clinic',
            '2A': 'A&E other provider',
            '31': 'Ante partum',
 '13': 'Planned',
 '22': 'GP',
 '23': 'Bed bureau',
 '2B': 'Transfer (emergency)',
 '28': 'Other',
 '11': 'Waiting list',
 '2D': 'Other emergency',
 '81': 'Transfer (non-emerg.)',
 '21': 'A&E'}


conspec = {
"100": "GENERAL SURGERY",
"101": "UROLOGY",
"110": "TRAUMA & ORTHOPAEDICS",
"120": "ENT",
"130": "OPHTHALMOLOGY",
"140": "ORAL SURGERY",
"141": "RESTORATIVE DENTISTRY",
"142": "PAEDIATRIC DENTISTRY",
"143": "ORTHODONTICS",
"145": "ORAL & MAXILLO FACIAL SURGERY",
"146": "ENDODONTICS",
"147": "PERIODONTICS",
"148": "PROSTHODONTICS",
"149": "SURGICAL DENTISTRY",
"150": "NEUROSURGERY",
"160": "PLASTIC SURGERY",
"170": "CARDIOTHORACIC SURGERY",
"171": "PAEDIATRIC SURGERY",
"180": "ACCIDENT & EMERGENCY",
#"191": "PAIN MANAGEMENT (Retired 1 April 2004)",
"190": "ANAESTHETICS",
"192": "CRITICAL CARE MEDICINE",
"300": "GENERAL MEDICINE",
"301": "GASTROENTEROLOGY",
"302": "ENDOCRINOLOGY",
"303": "CLINICAL HAEMATOLOGY",
"304": "CLINICAL PHYSIOLOGY",
"305": "CLINICAL PHARMACOLOGY",
"310": "AUDIOLOGICAL MEDICINE",
"311": "CLINICAL GENETICS",
"312": "CLINICAL CYTOGENETICS and MOLECULAR GENETICS (Retired 1 April 2010)",
"313": "CLINICAL IMMUNOLOGY and ALLERGY",
"314": "REHABILITATION",
"315": "PALLIATIVE MEDICINE",
"320": "CARDIOLOGY",
"321": "PAEDIATRIC CARDIOLOGY",
"325": "SPORT AND EXERCISE MEDICINE",
"326": "ACUTE INTERNAL MEDICINE",
"330": "DERMATOLOGY",
# "340": "RESPIRATORY MEDICINE (also known as thoracic medicine)",
"340": "RESPIRATORY MEDICINE",
"340": "RESPIRATORY MEDICINE",    
"350": "INFECTIOUS DISEASES",
"352": "TROPICAL MEDICINE",
"360": "GENITOURINARY MEDICINE",
"361": "NEPHROLOGY",
"370": "MEDICAL ONCOLOGY",
"371": "NUCLEAR MEDICINE",
"400": "NEUROLOGY",
"401": "CLINICAL NEURO-PHYSIOLOGY",
"410": "RHEUMATOLOGY",
"420": "PAEDIATRICS",
"421": "PAEDIATRIC NEUROLOGY",
"430": "GERIATRIC MEDICINE",
"450": "DENTAL MEDICINE SPECIALTIES",
"451": "SPECIAL CARE DENTISTRY",
"460": "MEDICAL OPHTHALMOLOGY",
"500": "OBSTETRICS and GYNAECOLOGY",
"501": "OBSTETRICS",
"502": "GYNAECOLOGY",
"504": "COMMUNITY SEXUAL AND REPRODUCTIVE HEALTH",
"510": "ANTENATAL CLINIC (Retired 1 April 2004)",
"520": "POSTNATAL CLINIC (Retired 1 April 2004)",
"560": "MIDWIFE EPISODE",
"600": "GENERAL MEDICAL PRACTICE",
"601": "GENERAL DENTAL PRACTICE",
"610": "MATERNITY FUNCTION (Retired 1 April 2004)",
"620": "OTHER THAN MATERNITY (Retired 1 April 2004)",
"700": "LEARNING DISABILITY",
"710": "ADULT MENTAL ILLNESS",
"711": "CHILD and ADOLESCENT PSYCHIATRY",
"712": "FORENSIC PSYCHIATRY",
"713": "PSYCHOTHERAPY",
"715": "OLD AGE PSYCHIATRY",
"800": "CLINICAL ONCOLOGY",
# "800": "CLINICAL ONCOLOGY (previously RADIOTHERAPY)",    
"810": "RADIOLOGY",
"820": "GENERAL PATHOLOGY",
"821": "BLOOD TRANSFUSION",
"822": "CHEMICAL PATHOLOGY",
"823": "HAEMATOLOGY",
"824": "HISTOPATHOLOGY",
"830": "IMMUNOPATHOLOGY",
"831": "MEDICAL MICROBIOLOGY AND VIROLOGY",
# "832": "NEUROPATHOLOGY (Retired 1 April 2004)",
"833": "MEDICAL MICROBIOLOGY (also known as MICROBIOLOGY AND BACTERIOLOGY)",
"834": "MEDICAL VIROLOGY",
"900": "COMMUNITY MEDICINE",
"901": "OCCUPATIONAL MEDICINE",
"902": "COMMUNITY HEALTH SERVICES DENTAL",
"903": "PUBLIC HEALTH MEDICINE",
"904": "PUBLIC HEALTH DENTAL",
"950": "NURSING EPISODE",
"960": "ALLIED HEALTH PROFESSIONAL EPISODE"
# "990": "JOINT CONSULTANT CLINICS (Retired 1 April 2004)",
}

morbidities = {"104": "Male genital disorders", #non cancer
               "105": "Female genital disorders", #non cancer
               "106": "Female reproductive disorders",
               "11" : "Cancer of rectum and anus",
               "128": "Complication of device, implant or graft",
#                "22" : "Male genital disorders", #cancer
#                "20" : "Female genital disorders", #cancer
               "19" : "Cancer of uterus",
               "2"  : "Septicaemia, shock",
               "26" : "Hodgkin's disease, Non-Hodgkin's lymphoma",
               "29" : "Cancer; chemotherapy; radiotherapy",
               "30" : "Secondary malignancies",
               "39" : "Deficiency and other anemia, Acute posthemorrhagic anemia",
               "4"  : "Mycoses",
               "40" : "Diseases of white blood cells",
               "42" : "Mental retardation, Senility and organic mental disorders",
               "54" : "Heart valve disorders",
               "56" : "Essential hypertension, Hypertension with complications and secondary hypertension",
               "58" : "Coronary atherosclerosis and other heart disease",
               "70" : "Aortic and peripheral arterial embolism or thrombosis",
               "74" : "Acute bronchitis",
               "75" : "Chronic obstructive pulmonary disease and bronchiectasis",
               "78" : "Pleurisy; pneumothorax; pulmonary collapse",
               "79" : "Respiratory failure; insufficiency; arrest (adult)",
               "3"  : "Bacterial infection; unspecified site", 
               "107": "Skin and subcutaneous tissue infections",
               "119": "Other perinatal conditions",
               "42" : "Mental retardation, Senility and organic mental disorders",
               "68" : "Peripheral and visceral atherosclerosis",
               "83" : "Intestinal infection",
               "88" : "Regional enteritis and ulcerative colitis"
              }


# Anemia & Deficiency and other anemia, Acute posthemorrhagic anemia \\
# Arterial diseases & Aortic and peripheral arterial embolism or thrombosis\\
# Atherosclerosis & Peripheral and visceral atherosclerosis\\
# Bronchitis (acute) & Acute bronchitis\\
# Bronchitis (chronic) & Chronic obstructive pulmonary disease and bronchiectasis\\
# Cancer (rectum) & Cancer of rectum and anus\\
# Cancer (secondary) & Secondary malignancies\\
# Cancer (therapy) & Cancer; chemotherapy; radiotherapy\\
# Cancer (uterus) & Cancer of uterus\\
# Coronary diseases & Coronary atherosclerosis and other heart diseases\\
# Enteritis/colitis & Regional enteritis and ulcerative colitis\\
# Genital disorders (F)& Female genital disorders\\
# Genital disorders (M)& Male genital disorders\\
# Heart-valve disorders& Heart valve disorders\\
# Hypertension & Essential hypertension, Hypertension with complications and secondary hypertension\\
# Implant/graft & Complication of device, implant or graft\\
# Infection (intestinal) & Intestinal infection\\
# Infection (skin) & Skin and subcutaneous tissue infections\\
# Infection (unspecified)& Bacterial infection; unspecified site\\
# Lung disorders & Pleurisy; pneumothorax; pulmonary collapse\\
# Lymphoma &Hodgkin’s disease, Non-Hodgkin’s lymphoma\\
# Mental disorders& Mental retardation, Senility and organic mental disorders\\
# Mycoses & Mycoses\\
# Reprod. disorders (F) & Female reproductive disorders\\
# Respiratory insuffic. & Respiratory failure; insufficiency; arrest (adult)\\
# Septicaemia & Septicaemia, shock\\
# White-cell diseases & Diseases of white blood cells\\
# CF & Cystic fibrosis
morbidities = {"104": "Genital disorders (M)", #non cancer
               "105": "Genital disorders (F)", #non cancer
               "106": "Reprod. disorders (F)",
               "11" : "Cancer (rectum)",
               "128": "Implant/graft",
               "19" : "Cancer (uterus)",
               "2"  : "Septicaemia",
               "26" : "Lymphoma",
               "29" : "Cancer (therapy)",
               "30" : "Cancer (secondary)",
               "39" : "Anemia",
               "4"  : "Mycoses",
               "40" : "White-cell diseases",
               "42" : "Mental disorders",
               "54" : "Heart-valve disorders",
               "56" : "Hypertension",
               "58" : "Coronary diseases",
               "70" : "Arterial diseases",
               "74" : "Bronchitis (acute)",
               "75" : "COPD/bronchiectasis",
               "78" : "Lung disorders",
               "79" : "Respiratory insuffic.",
               "3"  : "Infection (unspecified)", 
               "107": "Infections (skin)",
               "119": "Perinatal conditions",
               "42" : "Mental disorders",
               "68" : "Atherosclerosis",
               "83" : "Infection (intestinal)",
               "88" : "Enteritis/colitis"
              }


def number_of_comorbidities(row):
    '''
    Returns the number of values in fields SDG1, SDG2, SDG3, etc., that are > than zero.
    '''
    SDGs = ['SDG1', 'SDG2', 'SDG3', 'SDG4', 'SDG5', 'SDG6', 'SDG7', 'SDG8', 'SDG9', 'SDG10', 'SDG11', 'SDG12', 'SDG13'] 
    return (row[SDGs] != 'nan').sum()



def fill_new_field(row, SDG):    
    if SDG in [s.split('.')[0] for s in row.values.astype(str)]:
        return True
    else:
        return False

def stratify_SDGs_from_dict(data):
    df = data.copy()
    SDGs = ['SDG1', 'SDG2', 'SDG3', 'SDG4', 'SDG5', 'SDG6', 'SDG7', 'SDG8', 'SDG9', 'SDG10', 'SDG11', 'SDG12', 'SDG13'] 
    for key in list(morbidities.keys()):
        df['SDG_' + key] = df[SDGs].apply(fill_new_field, args=(key,), axis=1)
    df.drop(SDGs, axis=1, inplace=True)
#     df.drop(['DIAG1','DIAG2', 'DIAG3', 'DIAG4', 'DIAG5', 'DIAG6', 'DIAG7', 'DIAG8', 'DIAG9', 'DIAG10', 'DIAG11', 'DIAG12', 'DIAG13'],
#             axis=1, inplace=True)
    return df

def stratify_all_SDGs(data):
    df = data.copy()
    SDGs = ['SDG1', 'SDG2', 'SDG3', 'SDG4', 'SDG5', 'SDG6', 'SDG7', 'SDG8', 'SDG9', 'SDG10', 'SDG11', 'SDG12', 'SDG13']
    keys = [str(s).split('.')[0] for s in data[SDGs].values.flatten()]
    keys = list(set(keys))
    keys = [k for k in keys if not (str(k) == 'nan')]
    for key in keys:
        df['SDG_' + key] = df[SDGs].apply(fill_new_field, args=(key,), axis=1)
    df.drop(SDGs, axis=1, inplace=True)
#     df.drop(['DIAG1','DIAG2', 'DIAG3', 'DIAG4', 'DIAG5', 'DIAG6', 'DIAG7', 'DIAG8', 'DIAG9', 'DIAG10', 'DIAG11', 'DIAG12', 'DIAG13'],
#             axis=1, inplace=True)
    return df



def stratify_ADMIMETH(data):
    df = data.copy()
    keys = [str(s) for s in data.ADMIMETH.unique()]
    keys = list(set(keys))
    for key in keys:
        df['ADMIMETH_' + key] = df.ADMIMETH.apply(lambda x, key: True if str(x)==key else False, args=(key, ))
    df.drop('ADMIMETH', axis=1, inplace=True)
    return df
    
    
def stratify_CONSPEF(data):
    df = data.copy()
    keys = [str(s) for s in data.CONSPEF.unique()]
    keys = list(set(keys))
    for key in keys:
        df['CONSPEF_' + key] = df.CONSPEF.apply(lambda x, key: True if str(x)==key else False, args=(key, ))
    df.drop('CONSPEF', axis=1, inplace=True)
    return df

    
def stratify_ABX(data):
    df = data.copy()
    keys = [str(s) for s in data.ABX.unique()]
    keys = list(set(keys))
    for key in keys:
        df['ABX_' + key] = df.ABX.apply(lambda x, key: True if str(x)==key else False, args=(key, ))
    df.drop('ABX', axis=1, inplace=True)
    return df


    
def stratify_ORG(data, ORG_column, keys):
    df = data.copy()
    keys = [str(s) for s in keys]
    keys = list(set(keys))
    for key in keys:
        df['ORG_' + key] = df[ORG_column].apply(lambda x, key: True if str(x)==key else False, args=(key, ))
    df.drop(ORG_column, axis=1, inplace=True)
    return df



def code_to_diagnosis(code):
    import stratify as st
    x = code.split('_')
    if x[0] == 'SDG':
        return 'diagnosis: ' + st.morbidities.get(x[1])
    elif x[0] == 'CONSPEF':
        tmp = st.conspec.get(x[1])
        if tmp is None:
#             return 'CONSPEF not known'
            return 'Cons. specialty not known'
        else:
            return 'CONSPEF ' + st.conspec.get(x[1])
    elif x[0] == 'ADMIMETH':
        return 'admim. ' + st.admimeth.get(x[1])
    elif 'prev_prescr' in code:
        return x[0] + ' previously prescribed'
    elif 'r_prev' in code:
        return 'resistance to ' + x[0] + ' previously found'
    elif 's_prev' in code:
        return 'susceptibility to ' + x[0] + ' previously found'
    elif code == 'ABX_b96':
        return 'ABX given in 96 hrs'
    elif 'ABX_b72' in code:
        return x[2] + ' early prescription'
    elif code == 'n_comorb':
        return 'No. como.'
    elif code == 'n_days':
        return 'number of days'
    elif code == 'start_from_admi':
        return 'interval between admission and ABX administration'
    elif code == 'interval':
        return 'interval between admission and laboratory test'
    elif code == 'fraction_days_in_hospitals':
        return 'Frac. time in hosp.'
    elif code == 'is_M':
        return 'M'
    elif code == 'R_b72':
#         return 'tested in 72 hrs'
        return 'Early test'
    elif code == 'number_of_hospital_visits':
        return '# of admissions'
    else:
        return ' '.join(x)
    
    

def code_to_diagnosis2(code):
    x = code.split('_')
    if x[0] == 'SDG':
        return 'diagnosis: ' + st.morbidities.get(x[1])
    elif x[0] == 'CONSPEF':
        tmp = st.conspec.get(x[1])
        if tmp is None:
            return 'CONSPEF not known'
        else:
            return 'CONSPEF ' + st.conspec.get(x[1])
    elif x[0] == 'ADMIMETH':
        return 'admim. ' + st.admimeth.get(x[1])
    elif 'prev_prescr' in code:
        return x[0] + ' previously prescribed'
    elif 'r_prev' in code:
        return 'Previously tested in ' + x[0] + ' found R'
    elif 's_prev' in code:
        return 'Previously tested in ' + x[0] + ', found S'
    elif code == 'ABX_b96':
        return 'ABX given in 96 hrs'
    elif 'ABX_b72' in code:
        return x[2] + ' prescribed within 72hrs'    
    elif code == 'n_comorb':
        return 'number of comorbidities'
    elif code == 'n_days':
        return 'number of days'
    elif code == 'start_from_admi':
        return 'interval between admission and ABX administration'
    elif code == 'interval':
        return 'interval between admission and laboratory test'
    elif code == 'R_b72':
        return 'tested in 72 hrs'
    elif x[0] == 'ABX':
        return "Currently treated with " + x[1]
    else:
        return ' '.join(x)


vcode_to_diagnosis_v = np.vectorize(code_to_diagnosis, otypes=[str])


def stratify_early_prescription(data, early_string='ABX_b72'):
    df = data.copy()
    keys = [str(s) for s in data.ABX.unique()]
    keys = list(set(keys))
    for key in keys:
        df[early_string + '_' + key] = df.apply(lambda x, key: True if (str(x.ABX)==key) and (x[early_string]) else False, args=(key, ), axis=1)
    df.drop([early_string], axis=1, inplace=True)
    return df

def stratify_early_prescription2(data, early_string='ABX_b72'):
    df = data.copy()
    keys = ['ABX_ETP', 'ABX_CTX', 'ABX_AK', 'ABX_CIP', 'ABX_TAZ', 'ABX_ATM', 'ABX_GT', 'ABX_TEM', 'ABX_MEM', 'ABX_AUG',
       'ABX_CAZ']
    for key in keys:
        df[early_string + '_' + key.split('_')[1]] = df.apply(lambda x, key: True if (x[key]) and (x[early_string]) else False, args=(key, ), axis=1)
    df.drop([early_string], axis=1, inplace=True)
    return df


def date2int(x):
    Min = x.min()
    tmp = x - x.min()
    print("First date is ",Min)
    return tmp.dt.days



def int2date(x, Min):
    x = x.apply(lambda _: pd.Timedelta(days=_))
    return x + pd.Timestamp(Min)




