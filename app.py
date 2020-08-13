from flask import Flask, render_template, request
import pandas as pd
import csv
import os 
import urllib.request
import re

import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET','POST'])
def data():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        df = pd.read_csv(uploaded_file.filename)
        os.remove(str(uploaded_file.filename))
        df.to_csv('data.csv')

    
    return render_template('index.html')

#########################################################################################

# change the predict function and add a html page to show the predicted output
@app.route('/predict',methods=['POST'])
def predict():
    main_l=[]
    if(os.path.isfile('./data.csv')):
        df=pd.read_csv('data.csv')
        
        data=df
        stop = ['%HESITATION',"%HESITATION."]
        data['clean_text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        doc_data = data[data['speaker'] == 'Doctor']
        doc_text = doc_data['clean_text'].str.cat(sep=' ')
        pat_data = data[data['speaker'] == 'Patient']
        pat_text = pat_data['clean_text'].str.cat(sep=' ')

    # Separating the Doctor and Patient conversation. 
        pat_data = data[data['speaker'] == 'Patient']
        pat_text = pat_data['clean_text'].str.cat(sep=' ')

        '''
        def make_bold(item_list, line_list):
            for i in item_list:
            for idx,j in enumerate(line_list):
                if " "+i+" " in j: #for highlighting middle words
                    j = j.replace(" "+i+" ",' '+'\033[94m'+'\033[1m'+'\033[4m'+str(i).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'+' ')
                    line_list[idx]=j
                if " "+i+"." in j: #for highlighting end words
                    j = j.replace(" "+i+".",' '+ '\033[94m'+'\033[1m'+'\033[4m'+str(i).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'+' ')
                    line_list[idx]=j
                if j[0]==i[0] and i+" " in j: #for highlighting beginning words
                    j = j.replace(i+" ",' '+'\033[94m'+'\033[1m'+'\033[4m'+str(i).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'+' ')
                    line_list[idx]=j
        return line_list
        '''

    # EXTRACTING PATIENT COMPLAINTS depending on diseases and organs
        final_diseases_list = ['nausea','heart attack','abnormal heart rhythm','abnormal heart rate','heart flip flop','heart racing','liver damage','post diabeties','post diabetic','pre diabeties','pre diabetic','fatigue','breath shortness','congestive heart failure','heart failure','elevated cholesterol','elevated homocysteine','major blockage','medium cholesterol','high cholesterol','low cholesterol','very low cholesterol','very high cholesterol','heart issues','heart problems','bypass surgery','palpitation', 'cholesterol','back injury', 'prostate cancer', 'bagel syncope', 'redness swelling pain', 'outflow tract obstruction', 'preeclampsia','mercury', 'iron', 'PAC', 'cardiomyopathy', 'restless', 'homocysteine', 'shortness of breath', 'metabolic syndrome', 'aneurysms', 'diabetic', 'cough', 'vitamin D', 'gastric ulcer', 'pain heaviness', 'chest pain', 'alcohol', 'seizure', 'pains', 'psoriasis', 'magnesium vitamin', 'premature ventricular contractions', 'sinus infections', 'penicillin', 'pulmonary embolism', 'cholesterol diabetes', 'Lipitor', 'diabetes your kidneys', 'NAC', 'vitamin B.', 'familial heart disease', 'shortness of breath.', 'heart aches', 'respiratory tract infection', 'familial hyperlipidemia', 'cardiovascular disease', 'headedness', 'coronary disease', 'deaths', 'arthritis', 'dyslipidemia', 'premature atrial contraction', 'burns', 'hepatitis', 'calcium', 'mini stroke', 'overdosed', 'herniated', 'aneurysm', 'AKG', 'tunnel vision', 'congenital', 'borderline', 'Mencia', "Alzheimer's", 'shortness of breath', 'snoring', 'muscle aches', 'psoriatic', 'cardiac block', 'skin cancer', 'numb', 'muscle cramps', 'chest discomfort shortness of breath lightheadedness', 'oxygen', 'allergy', 'bruises', 'coronary artery disease', 'curry', 'smoking', 'dementia', 'anemia', 'head injury', 'premature coronary disease', 'infection', 'weight loss', 'sleep apnea', 'obesity', 'borderline hypertrophy', 'spasm', 'Nicene', 'automatic dysfunction', 'shortness of breath', 'depressed', 'hypertrophy', 'stroke', 'prolapse syndrome', 'tears', 'heart pounding','pain', 'torture', 'congestive heart failure', 'inflammation', 'prolapse', 'tachycardia', 'grief', 'iron deficiency', 'bleeding', 'scoliosis', 'heart palpitations','breast cancer', 'testosterone','swelling', 'heartburn', 'triglycerides','dry mouth', 'bronchitis', 'atherosclerosis', 'moisture','heart murmur ', 'syncope', 'hypothermia', 'seizures', 'Stephens', 'bad headaches', 'sores', 'murmur', 'heart failure','myelofibrosis', 'glucose','torsion', 'sore','skipping beats sensation', 'shortness of breath light headedness', 'lightheadedness', 'sinus tachycardia', 'fatigue shortness of breath', 'infections', 'aneurysm', 'inherited disorders', 'muscle issues like shoulder', 'shortness of breath disorientation', 'obstructive disease', 'pericarditis', 'alcohol mistreating', "Turner's", 'coronary artery calcification', 'cramps', 'varicose', 'nail', 'chest tightness', 'tingling', 'dehydration', 'premature coronary diseases familial', 'magnesium iron', 'heart disease', 'fever', 'strokes', 'vitamin D.', 'blood disorders', 'sneeze', 'vitamin', 'cardiac death', 'prediabetes', 'obstructive lesion', 'failure', 'weakness', 'diabetes', 'coronary disease friends', 'sinus infection', 'heart murmur', 'palpitations', 'genetic disorder', 'ADD', "Raynaud's", 'coronary calcification', 'diabetes', 'emphysema', 'diabetic coma', 'anxiety', 'heart failure cardiomyopathy', 'shortness of breath','general disorientation', 'dizziness', 'headache', 'brain tumor','tumor', 'bruising', 'loss', 'liver damage', 'chest pains', 'shock', 'homicide', 'premature coronary disease familial', 'panic', 'vascular disease', 'poisoning', 'clotting disorders', 'cancer', 'vertigo', 'panic attacks', 'kidney stones', 'appetite', 'HA', 'nicotine', 'kidney infections', 'heart attacks', 'sleep disorders', 'allergic', 'diabetes stroke', 'low carb eating', 'trauma', 'allergic reaction', 'burn', 'fainting', 'atrial septal defect', 'shortness of breath and stuff', 'hypertension', 'sweats', 'cramping', 'constipation', 'insomnia', 'vitamin C', 'muscular probable Musco skeletal pain', 'familial', 'delusions', 'bad reflux', 'hysterectomy', 'death', 'GI tract reflux', 'obstructive coronary disease', 'exertional', 'prolapse prolapse', 'sick anxiety', 'cholesterol disorder', 'thyroid disorders', 'calcification', 'headaches', 'apnea', 'herpes', 'shortness of breath nothing', 'sneezing', 'fatty liver', 'colon cancer']
        # Dynamic list of diseases
        pat_d_f_list = final_diseases_list

    #making a list of organs
        organs_list=['Mouth', 'head','lower back', 'my back', 'leg', 'Teeth', 'Tongue', 'Salivary glands', 'Parotid glands', 'Submandibular glands', 'Sublingual glands', 'Larynx', 'Esophagus', 'Stomach', 'Small intestine', 'Duodenum', 'Jejunum', 'Ileum', 'Large intestine', 'Liver', 'Gallbladder', 'Mesentery', 'Pancreas', 'Anal canal and anus', 'Blood cells', 'Respiratory system', 'Nasal cavity', 'Pharynx', 'Larynx', 'Trachea', 'Bronchi', 'Lungs', 'Diaphragm', 'Urinary system', 'Kidneys', 'Ureter', 'Bladder', 'Urethra', 'Reproductive organs', 'Female reproductive system', 'Internal reproductive organs', 'Ovaries', 'Fallopian tubes', 'Uterus', 'Vagina', 'External reproductive organs', 'Vulva', 'Clitoris', 'Placenta', 'Male reproductive system', 'Internal reproductive organs', 'Testes', 'Epididymis', 'Vas deferens', 'Seminal vesicles', 'Prostate', 'Bulbourethral glands', 'External reproductive organs', 'Penis', 'Scrotum', 'Endocrine system', 'Pituitary gland', 'Pineal gland', 'Thyroid gland', 'Parathyroid glands', 'Adrenal glands', 'Pancreas', 'Circulatory system', 'Circulatory system', 'Heart', 'Patent Foramen Ovale', 'Arteries', 'Veins', 'Capillaries', 'Lymphatic system', 'Lymphatic vessel', 'Lymph node', 'Bone marrow', 'Thymus', 'Spleen', 'Gut-associated lymphoid tissue', 'Tonsils', 'Interstitium', 'Nervous system', 'Brain', 'Cerebrum', 'Cerebral hemispheres', 'Diencephalon', 'The brainstem', 'Midbrain', 'Pons', 'Medulla oblongata', 'Cerebellum', 'The spinal cord', 'The ventricular system', 'Choroid plexus', 'Peripheral nervous system', 'Nerves', 'Cranial nerves', 'Spinal nerves', 'Ganglia', 'Enteric nervous system', 'Sensory organs', 'Eye', 'Cornea', 'Iris', 'Ciliary body', 'Lens', 'Retina', 'Ear', 'Outer ear', 'Earlobe', 'Eardrum', 'Middle ear', 'Ossicles', 'Inner ear', 'Cochlea', 'Vestibule of the ear', 'Semicircular canals', 'Olfactory epithelium', 'Tongue', 'Taste buds', 'Integumentary system', 'Mammary glands', 'Skin', 'Subcutaneous tissue']

    #extracting complaints = sentences containing diseases + sentences containing organ mentions
        disease_complaints=[]
        organ_complaints=[]
        patient_complaints = []     #disease_complaints + organ_complaints
        
        patient_diseases = []
        mentioned_organs=[]

        for line in pat_text.split('.'):
            line= str(line)+"."
            for i in pat_d_f_list:
                if " "+i+" " in line or " "+i+"." in line or  (line[0]==i[0] and i+" " in line):
                    if line not in disease_complaints:
                        disease_complaints.append(line)
                    if line not in patient_complaints:
                        patient_complaints.append(line)
                    patient_diseases.append(i)
            for i in organs_list:
                i=i.lower()
                if " "+i+" " in line.lower() or " "+i+"." in line.lower() or  (line[0].lower==i[0] and i+" " in line.lower()):
                    mentioned_organs.append(i)
                    if line not in organ_complaints:
                        organ_complaints.append(line)
                    if line not in patient_complaints:
                        patient_complaints.append(line)

                
    #removing repeated mentions
        complaints = list(set(patient_complaints))
    #converting complaints to lower case so that Capital initials won't cause problem
        for idx, line in enumerate(complaints):
            complaints[idx] = line.lower()
        
        patient_diseases=(list(set(patient_diseases)))
        mentioned_organs=(list(set(mentioned_organs)))
        disease_complaints = list(set(disease_complaints))
        organ_complaints = list(set(organ_complaints))



    #get patient family history

    #list of possessive pronouns
        pronouns_list=['whom','whose','her','my','your','his','its','hers','mine','their','our','ours','theirs']
    #list of relational pronouns
        relation_list=['ancestors', 'mom','aunt', 'child', 'uncle', 'father', 'niece', 'nephew', 'baby', 'bachelor', 'boy', 'wife', 'husband', 'partner', 'bride', 'bridegroom', 'brother', 'brother-in-law', 'child', 'cousin', 'daughter', 'daughter-in-law', 'descendants', 'elder brother', 'elder sister', 'family', 'father-in-law', 'female', 'foster-brother', 'foster-child', 'foster-father', 'foster-mother', 'foster-sister', 'girl', 'grand-child', 'grand-children', 'grand-daughter', 'grand-father', 'grand-mother', 'grand-son', 'great grand son', 'infant', 'lad', 'lover', 'maid', 'master', 'mother', 'parents', 'parent', 'sister', 'sister-in-law', 'son', 'son-in-law', 'spouse', 'step brother', 'step daughter', 'step father', 'step mother', 'step sister', 'younger brother', 'younger sister', 'younger daughter', 'younger son', 'elder daughter', 'elder son']


        
    #extracting patient family history
        disease_history=[]
        diseases_in_family=[]

        for line in complaints:
            tokns=re.split('; | |, |\*|\n',line)
            for idx,i in enumerate(tokns):
                if i in pronouns_list and (tokns[idx+1] in relation_list  or tokns[idx+1][:-1] in relation_list):
                    reln=""
                    z=2
                    reln=reln+" "+str(i)+" "+str(tokns[idx+1])
                    while (tokns[idx+z] == "'s"):
                        reln=reln+tokns[idx+z]+" "+tokns[idx+z+1]
                        z=z+1
                    for j in pat_d_f_list:
                        if j in line:
                            disease_history.append(line)
                            diseases_in_family.append(j)
                            
                            

    # Removing patient history from patient complaints

        for i in disease_history:
            if i in disease_complaints:
                disease_complaints.remove(i)
            if i in organ_complaints:
                organ_complaints.remove(i)

        all_lines= ""
        for line in disease_complaints:
            all_lines= all_lines+" "+line

        for i in patient_diseases:
            if i not in all_lines:
                patient_diseases.remove(i)
            
        all_lines= ""
        for line in organ_complaints:
            all_lines= all_lines+" "+line
        
        for i in mentioned_organs:
            if i not in all_lines:
                mentioned_organs.remove(i)
            
            
    #extracting disease / organ enquiries by doctor

        yes_no_list=['yep', 'Yes', 'nope', 'Yeah', 'I do',  'i do have', 'yo', 'yes', 'yeah', 'no', 'Yep', 'do not', 'nop', 'nah', "don't"]
        no_list=['nope','no','do not', 'nop', 'nah', "don't"]
        yes_list=['yep', 'Yes',  'Yeah', 'I do',  'i do have', 'yo', 'yes', 'yeah',  'Yep', ]

        orgns=[]
        diseasess=[]
        disease_q_a=[]
        orgns_q_a=[]
        for idx,i in enumerate(data['clean_text']):
        
        #checking if doctor mentions any diseases
            if (data['speaker'][idx]=="Doctor"):
                for j in final_diseases_list:
                    if " "+j+" " in i or " "+j+"." in i or j+" " in i:
                        q_a_disease=[{data['speaker'][idx]:data['clean_text'][idx]}]
                    #checking if the patient response contains yes/no words
                        for yn in yes_list:
                            if " "+yn+" " in data['clean_text'][idx+1] or " "+yn+"." in data['clean_text'][idx+1] or (data['clean_text'][idx+1][0]==yn[0] and yn+" " in data['clean_text'][idx+1]):
                                if j not in diseasess:
                                    diseasess.append(j)
                                q_a_disease.append([{data['speaker'][idx+1]:data['clean_text'][idx+1]}])
                                if (q_a_disease) not in disease_q_a:
                                    disease_q_a.append(q_a_disease)
                            
        #checking if doctor mentions any organs
            if (data['speaker'][idx]=="Doctor"):
                for j in organs_list:
                    if " "+j+" " in i.lower() or " "+j+"." in i.lower() or j+" " in i.lower():
                        q_a_organ=[{data['speaker'][idx]:data['clean_text'][idx]}]
                    
                    #checking if the patient response contains yes/no words
                        for yn in yes_list:
                            if " "+yn+" " in data['clean_text'][idx+1] or " "+yn+"." in data['clean_text'][idx+1] or (data['clean_text'][idx+1][0]==yn[0] and yn+" " in data['clean_text'][idx+1]):
                                if j not in orgns:
                                    orgns.append(j)
                                q_a_organ.append([{data['speaker'][idx+1]:data['clean_text'][idx+1]}])
                                if (q_a_organ) not in orgns_q_a:
                                    orgns_q_a.append(q_a_organ)
                                break
                    
        diseasess=list(set(diseasess))
        orgns=list(set(orgns))



    # Extracting food mentions

        food_list=['spinach','berries','brocolli','mediterranean diet','avocado nuts','water', 'chickens', 'ingredients', 'oats', 'rice quinoa', 'hydrochloric thiazide', 'glucose', 'Motrin', 'tobacco products', 'product', 'potassium', 'mutations', 'mutation', 'lipid', 'breakfast', 'triglyceride', 'proteins', 'aspirin', 'channel', 'meats', 'fruits', 'hemoglobin', 'triglycerides', 'vegan', 'magnesium', 'caffeine', 'tobacco', 'platelets', 'eggs', 'CRP I', 'based products', 'hormone', 'rice', 'fatty food', 'non-veg', 'magnesium vitamin', 'proteins carbohydrates', 'wheat', 'animal protein', 'plant', 'pharmacies', 'radish', 'thiazide diuretic', 'animals products', 'flaxseeds', 'vitamin C', 'unsaturated', 'LDL', 'quinoa', 'fish', 'nutrients', 'substance', 'hydrochloric', 'steroids', 'acid', 'alcohol', 'meclizine', 'enzymes', 'vitamin', 'honey', 'salt', 'CRP', 'virus', 'protein', 'wine', 'diuretic', 'products', 'iron', 'coffee', 'calcium', 'drug', 'fruit', 'animal', 'aspirin I', 'vegetable', 'animals', 'magnesium I', 'vitamin D.', 'meat products', 'food', 'vegies', 'penicillin', 'nutrient', 'animal products', 'inflammatory', 'fatty', 'multivitamin', 'magnesium iron', 'salts', 'chicken', 'antibiotics', 'proteins plant', 'hormones', 'snacks', 'lunch', 'amino', 'pea', 'foods', 'atorvastatin', 'meat', 'allergic', 'pro inflammatory type', 'calcium channel', 'carbohydrates', 'lime', 'carbon', 'meals', 'dinner', 'non-vegetarian', 'vitamins', 'vegetables', 'drugs', 'platelet', 'thiazide', 'insulin', 'Lupron']
        food_rel_words=['eat','eating','diet','intake','consume','consuming','dieting','foods','eatables',]
        yes_no_list=['yep', 'Yes', 'nope', 'Yeah', 'I do', 'okay', 'i do have', 'yo', 'yes', 'yeah', 'ok', 'no', 'Yep', 'do not', 'nop', 'nah', "don't"]
        no_list=['nope','no','do not', 'nop', 'nah', "don't"]
        yes_list=['yep', 'Yes',  'Yeah', 'I do', 'okay', 'i do have', 'yo', 'yes', 'yeah', 'ok',  'Yep', ]

        food_mentioned=[] 
        food_sentences1=[]    #using food_rel_words list    
        food_sentences2=[]  #using foods list

    # extracting food sentences using food related words
        '''for idx,i in enumerate(data['clean_text']):
            i=i.lower()
            for x in food_rel_words:
                x=x.lower()
                if ((" "+x+" ").lower() in i ) or ((" "+x+".").lower() in i ):
                    food_sentences1.append(i)
                if ((x+" ") in i and i and x[0]==i[0] ):
                    food_sentences1.append(i)'''

    # extracting food sentences using food list
        for i in range(len(data['clean_text'])):
            data['clean_text'][i]=data['clean_text'][i].lower()
        #print(data['clean_text'][i],data['clean_text'][i][0])
            for x in food_list:
                a=x.lower()[0]
                if ((" "+x+" ").lower() in data['clean_text'][i]) or ((" "+x+".").lower() in data['clean_text'][i]) or ((x+" ").lower() in data['clean_text'][i] and data['clean_text'][i][0]==a ):
                    if x not in food_mentioned:
                        food_mentioned.append(x)
                    food_sentences2.append(data['clean_text'][i])

    #check foods items in sentences in food_sentences1
        for i in food_list:
            if " "+i+" ".lower() in food_sentences1 and " "+i+" ".lower() not in food_mentioned:
                food_mentioned.append(i)
            

    #final food sentences list, combining food_sentences1 and food_sentences2
        food_sentences=food_sentences1
        [food_sentences.append(x) for x in food_sentences2 if x not in food_sentences] 

        food_sentences=list(set(food_sentences))

    # Extracting food responses
        food_response=[]    #food responses 
        for idx,i in enumerate(data['clean_text']):
            i=i.lower()
            for x in food_sentences:
                if i==x:
                    for yn in yes_no_list:
                        if (" "+yn+" ".lower() in data['clean_text'][idx+1].lower()) or (yn+" ".lower() in data['clean_text'][idx+1].lower()) or (" "+yn+".".lower() in data['clean_text'][idx+1].lower()):
                            resp=  [ { data['speaker'][idx]:data['clean_text'][idx], data['speaker'][idx+1]:data['clean_text'][idx+1] }]
                            if resp not in food_response:
                                food_response.append(resp)


                            
        patient_diseases=list(reversed(sorted(patient_diseases,key=len)))
        updated_list=[]
        for k in range(len(disease_complaints)):
        #print(disease_complaints[k])
            tmp=disease_complaints[k]
        #print('tmp',tmp)
            for i in patient_diseases:   
                if i in tmp:
                #print(i)
                    updated_list.append(i)
                    tmp=tmp.replace(i,'')
                
        #print(i,tmp)
    #updated_list


        diseases_in_family=list(reversed(sorted(diseases_in_family,key=len)))
        updated_list1=[]
        for k in range(len(disease_history)):
        #print(disease_complaints[k])
            tmp=disease_history[k]
        #print('tmp',tmp)
            for i in diseases_in_family:   
                if i in tmp:
                #print(i)
                    updated_list1.append(i)
                    tmp=tmp.replace(i,'')
                
        #print(i,tmp)
        updated_list1=list(set(updated_list1))
                
        disease_history=list(set(disease_history))
    #print('*___LIST OF diseases FROM PATIENT___*')
    #print(updated_list)
        '''
        summarized_dis=[]
    #print('\n*___LIST OF DISEASE COMPLAINTS FROM PATIENT___*')
        for i in range(len(disease_complaints)):
            text=str(disease_complaints[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=20,
                                        max_length=40,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_dis.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_dis.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
    #print('*___LIST OF organs FROM PATIENT___*')
    #print(mentioned_organs)
        summarized_orgs=[]
        for i in range(len(organ_complaints)):
            text=str(organ_complaints[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n", text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=20,
                                        max_length=40,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_orgs.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_orgs.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
        print("_______________________________________________________")
        '''
        '''
        
        summarized_hist=[]
        for i in range(len(disease_history)):
            text=str(disease_history[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_hist.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_hist.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)


        s=""
        summary_food=[]
        for i in range(len(food_sentences)):
            tex=str(food_sentences[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',tex)
            if(len(k)>70):
                t5_prepared_Text = "summarize: "+tex
            #print ("original text preprocessed: \n",tex)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=50,
                                        max_length=150,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summary_food.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summary_food.append(tex)
            #print('\n','"This has no summary"---',tex,'\n')

        summarized_disease=[]
        summarized_disease1=[]

        for i in range(len(disease_q_a)):
            text=str(disease_q_a[i][0]['Doctor'])
            text1=str(disease_q_a[i][1][0]['Patient'])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            k1=re.split(r'\s+',text1)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_disease.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_disease.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
            if(len(k1)>50):
                t5_prepared_Text = "summarize: "+text1
                print ("original text preprocessed: \n",text1)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_disease1.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_disease1.append(text1)
            #print('"This has no summary"',text1,'\n')
        #print(idx,i)

    #print("_______________________________________________________")


    #print("\n\n\nDiseases list:\n")
    #print(diseasess)

        for i in range(len(disease_q_a)):
            for j in diseasess:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in disease_q_a[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    summarized_disease[i]=summarized_disease[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_disease1[i]=summarized_disease1[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_disease[i]=summarized_disease[i].replace(" "+str(j)+"."," "+hlt+".")
                    summarized_disease1[i]=summarized_disease1[i].replace(" "+str(j)+"."," "+hlt+".")
        for i in range(len(disease_q_a)):
            for j in diseasess:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in disease_q_a[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    disease_q_a[i][0]['Doctor']=disease_q_a[i][0]['Doctor'].replace(" "+str(j)+" "," "+hlt+" ")
                    disease_q_a[i][0]['Patient']=disease_q_a[i][1][0]['Patient'].replace(" "+str(j)+" "," "+hlt+" ")
                    disease_q_a[i][0]['Doctor']=disease_q_a[i][0]['Doctor'].replace(" "+str(j)+"."," "+hlt+".")
                    disease_q_a[i][0]['Patient']=disease_q_a[i][1][0]['Patient'].replace(" "+str(j)+"."," "+hlt+".")
                
        

        summarized_orgdis=[]
        summarized_orgdis1=[]
        for i in range(len(orgns_q_a)):
            text=str(orgns_q_a[i][0]['Doctor'])
            text1=str(orgns_q_a[i][1][0]['Patient'])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            k1=re.split(r'\s+',text1)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
                print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_orgdis.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_orgdis.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
            if(len(k1)>50):
                t5_prepared_Text = "summarize: "+text1
            #print ("original text preprocessed: \n",text1)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_orgdis1.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_orgdis1.append(text1)
            #print('"This has no summary"',text1,'\n')
        #print(idx,i)

    #print("\n\nOrgans list:\n")
    #print(orgns)

        
    for i in range(len(orgns_q_a)):
        for j in orgns:
            if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in orgns_q_a[i]:
                hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                orgns_q_a[i][0]['Doctor']=orgns_q_a[i][0]['Doctor'].replace(" "+str(j)+" "," "+hlt+" ")
                orgns_q_a[i][0]['Patient']=orgns_q_a[i][1][0]['Patient'].replace(" "+str(j)+" "," "+hlt+" ")
                orgns_q_a[i][0]['Doctor']=orgns_q_a[i][0]['Doctor'].replace(" "+str(j)+"."," "+hlt+".")
                orgns_q_a[i][0]['Patient']=orgns_q_a[i][1][0]['Patient'].replace(" "+str(j)+"."," "+hlt+".")

        
        for i in range(len(orgns_q_a)):
            for j in orgns:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in orgns_q_a[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    summarized_orgdis[i]=summarized_orgdis[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_orgdis1[i]=summarized_orgdis1[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_orgdis[i]=summarized_orgdis[i].replace(" "+str(j)+"."," "+hlt+".")
                    summarized_orgdis1[i]=summarized_orgdis1[i].replace(" "+str(j)+"."," "+hlt+".")            
                
                

        summarized_foodr=[]
        summarized_foodr1=[]

        for i in range(len(food_response)):
            text=str(food_response[i][0]['Doctor'])
            text1=str(food_response[i][0]['Patient'])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            k1=re.split(r'\s+',text1)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_foodr.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_foodr.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
            if(len(k1)>50):
                t5_prepared_Text = "summarize: "+text1
            #print ("original text preprocessed: \n",text1)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            #summarized_foodr1.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_foodr1.append(text1)
            #print('"This has no summary"',text1,'\n')
        #print(idx,i)

        
    for i in range(len(food_response)):
        for j in food_list:
            if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in food_response[i]:
                hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                food_response[i][0]['Doctor']=food_response[i][0]['Doctor'].replace(" "+str(j)+" "," "+hlt+" ")
                food_response[i][0]['Patient']=food_response[i][0]['Patient'].replace(" "+str(j)+" "," "+hlt+" ")
                food_response[i][0]['Doctor']=food_response[i][0]['Doctor'].replace(" "+str(j)+"."," "+hlt+".")
                food_response[i][0]['Patient']=food_response[i][0]['Patient'].replace(" "+str(j)+"."," "+hlt+".")

        for i in range(len(food_response)):
            for j in food_list:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in food_mentioned[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    summarized_foodr[i]=summarized_foodr[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_foodr1[i]=summarized_foodr1[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_foodr[i]=summarized_foodr[i].replace(" "+str(j)+"."," "+hlt+".")
                    summarized_foodr1[i]=summarized_foodr1[i].replace(" "+str(j)+"."," "+hlt+".")            
        '''
                
        #print('KEYWORDS PRESENT IN WHOLE TRANSCRIPTION\n')
        transcription=[]

        #print('*___LIST OF DISEASES FROM PATIENT___*')
        #print(*updated_list,sep='\n')
        transcription.append(updated_list)
        for i in range(len(disease_complaints)):
            disease_complaints[i]=str(i+1)+'.'+disease_complaints[i]
        transcription.append(disease_complaints)
        #print('\n*___LIST OF ORGANS FROM PATIENT___*')
        transcription.append(mentioned_organs)
        for i in range(len(organ_complaints)):
            organ_complaints[i]=str(i+1)+'.'+organ_complaints[i]
        transcription.append(organ_complaints)

        #print(*mentioned_organs,sep='\n')
        #print("\n*___LIST OF DISEASES IN FAMILY___*")
        #print(*updated_list1,sep='\n')
        transcription.append(updated_list1)
        for i in range(len(disease_history)):
            disease_history[i]=str(i+1)+'.'+disease_history[i]
        transcription.append(disease_history)
        #print("\n*___LIST OF FOODs MENTIONED___*")
        
        transcription.append(food_mentioned)
        #print(*food_mentioned,sep='\n')
        for i in range(len(food_sentences)):
            food_sentences[i]=str(i+1)+'. '+food_sentences[i]
        
        transcription.append(food_sentences)
        

        '''print('NOTE\n')
    #highlighting the normal response

        disease_complaints = make_bold(updated_list,disease_complaints)
        print('\n*___LIST OF DISEASE COMPLAINTS FROM PATIENT___*')


    #highlighting the summarized response
        summarized_dis= make_bold(updated_list,summarized_dis)


        for i in range(len(disease_complaints)):
            print(summarized_dis[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')




        organ_complaints = make_bold(mentioned_organs,organ_complaints)


    #highlighting the summarized response
        summarized_orgs= make_bold(mentioned_organs,summarized_orgs)

        print('\n*___LIST OF ORGAN COMPLAINTS FROM PATIENT___*')

        for i in range(len(organ_complaints)):
            print(summarized_orgs[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')
        
        


        disease_history = make_bold(updated_list1,disease_history)


        #highlighting the summarized response
        summarized_hist= make_bold(updated_list1,summarized_hist)
        print("\n\n__**DISEASES History IN FAMILY**__")

        for i in range(len(disease_history)):
            print(summarized_hist[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')




        food_sentences = make_bold(food_mentioned,food_sentences)


    #highlighting the summarized response
        summary_food= make_bold(food_mentioned,summary_food)
        print("\nFood related sentences:\n")


        for i in range(len(food_sentences)):
            print(summary_food[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')


        print('\nDISCUSSIONS\n\n')

        print("\n\n*__Disease discussion between doctor and patient:")
        for i in range(len(disease_q_a)):

            print('Doctor :',summarized_disease[i],'\n')
        
            print('Patient :',summarized_disease1[i],'\n')
            print('\n---------------------------------------------------------------------------------\n')
        
        

        print("\n\n*__Organ discussion between doctor and patient:")
        for i in range(len(orgns_q_a)):
            print('Doctor :',summarized_orgdis[i],'\n')
            print('Patient :',summarized_orgdis1[i],'\n')
            print('\n---------------------------------------------------------------------------------\n')

        

        print("\n\n*__Food discussion between doctor and patient:")
        for i in range(len(food_response)):
            print('Doctor :',summarized_foodr[i],'\n')
            print('Patient :',summarized_foodr1[i],'\n')
            '''
        #print('\n---------------------------------------------------------------------------------\n')
            
       
        #print('\n---------------------------------------------------------------------------------\n')
        main_l = transcription
            

    else:
        main_l.append('The file is not uploaded!!')
        return render_template('view1.html', prediction_text='{}'.format(main_l[0]))

    return render_template('predict.html', prediction_text0= main_l[0] , prediction_text1= main_l[1] , prediction_text2= main_l[2] , prediction_text3= main_l[3] , prediction_text4=main_l[4] , prediction_text5=main_l[5] , prediction_text6=main_l[6], prediction_text7=main_l[7])

###############################################################################################

#########################################################################################
# change the view html page for python ginger method from youtube vedio

@app.route('/view',methods=['POST'])
def view():
    '''
    For rendering results on HTML GUI
    '''
    l=[]
    if(os.path.isfile('./data.csv')):
        df=pd.read_csv('data.csv')
        for i in range(len(df)):
            temp = str(df.iloc[i][-1])+" : "+str(df.iloc[i][-2])
            l.append(str(temp))

    else:
        l.append('The file is not uploaded!!')
        return render_template('view.html', prediction_text='{}'.format(l[0]))


    
    return render_template('view.html', prediction_text=l)
###############################################################################################


@app.route('/soap',methods=['POST'])
def soap():
    '''
    For rendering results on HTML GUI
    '''
    main_l=[]
    if(os.path.isfile('./data.csv')):
        df=pd.read_csv('data.csv')
        data=df
        stop = ['%HESITATION',"%HESITATION."]
        data['clean_text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        doc_data = data[data['speaker'] == 'Doctor']
        doc_text = doc_data['clean_text'].str.cat(sep=' ')
        pat_data = data[data['speaker'] == 'Patient']
        pat_text = pat_data['clean_text'].str.cat(sep=' ')

    # Separating the Doctor and Patient conversation. 
        pat_data = data[data['speaker'] == 'Patient']
        pat_text = pat_data['clean_text'].str.cat(sep=' ')

        '''
        def make_bold(item_list, line_list):
            for i in item_list:
            for idx,j in enumerate(line_list):
                if " "+i+" " in j: #for highlighting middle words
                    j = j.replace(" "+i+" ",' '+'\033[94m'+'\033[1m'+'\033[4m'+str(i).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'+' ')
                    line_list[idx]=j
                if " "+i+"." in j: #for highlighting end words
                    j = j.replace(" "+i+".",' '+ '\033[94m'+'\033[1m'+'\033[4m'+str(i).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'+' ')
                    line_list[idx]=j
                if j[0]==i[0] and i+" " in j: #for highlighting beginning words
                    j = j.replace(i+" ",' '+'\033[94m'+'\033[1m'+'\033[4m'+str(i).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'+' ')
                    line_list[idx]=j
        return line_list
        '''

    # EXTRACTING PATIENT COMPLAINTS depending on diseases and organs
        final_diseases_list = ['brain tumor','nausea','heart attack','abnormal heart rhythm','abnormal heart rate','heart flip flop','heart racing','liver damage','post diabeties','post diabetic','pre diabeties','pre diabetic','fatigue','breath shortness','congestive heart failure','heart failure','elevated cholesterol','elevated homocysteine','major blockage','medium cholesterol','high cholesterol','low cholesterol','very low cholesterol','very high cholesterol','heart issues','heart problems','bypass surgery','palpitation', 'cholesterol','back injury', 'prostate cancer', 'bagel syncope', 'redness swelling pain', 'outflow tract obstruction', 'preeclampsia','mercury', 'iron', 'PAC', 'cardiomyopathy', 'restless', 'homocysteine', 'shortness of breath', 'metabolic syndrome', 'aneurysms', 'diabetic', 'cough', 'vitamin D', 'gastric ulcer', 'pain heaviness', 'chest pain', 'alcohol', 'seizure', 'pains', 'psoriasis', 'magnesium vitamin', 'premature ventricular contractions', 'sinus infections', 'penicillin', 'pulmonary embolism', 'cholesterol diabetes', 'Lipitor', 'diabetes your kidneys', 'NAC', 'vitamin B.', 'familial heart disease', 'shortness of breath.', 'heart aches', 'respiratory tract infection', 'familial hyperlipidemia', 'cardiovascular disease', 'headedness', 'coronary disease', 'deaths', 'arthritis', 'dyslipidemia', 'premature atrial contraction', 'burns', 'hepatitis', 'calcium', 'mini stroke', 'overdosed', 'herniated', 'aneurysm', 'AKG', 'tunnel vision', 'congenital', 'borderline', 'Mencia', "Alzheimer's", 'shortness of breath', 'snoring', 'muscle aches', 'psoriatic', 'cardiac block', 'skin cancer', 'numb', 'muscle cramps', 'chest discomfort shortness of breath lightheadedness', 'oxygen', 'allergy', 'bruises', 'coronary artery disease', 'curry', 'smoking', 'dementia', 'anemia', 'head injury', 'premature coronary disease', 'infection', 'weight loss', 'sleep apnea', 'obesity', 'borderline hypertrophy', 'spasm', 'Nicene', 'automatic dysfunction', 'shortness of breath', 'depressed', 'hypertrophy', 'stroke', 'prolapse syndrome', 'tears', 'heart pounding','back pain', 'torture', 'congestive heart failure', 'inflammation', 'prolapse', 'tachycardia', 'grief', 'iron deficiency', 'bleeding', 'scoliosis', 'heart palpitations','breast cancer', 'testosterone','swelling', 'heartburn', 'triglycerides','dry mouth', 'bronchitis', 'atherosclerosis', 'moisture','heart murmur ', 'syncope', 'hypothermia', 'seizures', 'Stephens', 'bad headaches', 'sores', 'murmur', 'heart failure','myelofibrosis', 'glucose','torsion', 'sore','skipping beats sensation', 'shortness of breath light headedness', 'lightheadedness', 'sinus tachycardia', 'fatigue shortness of breath', 'infections', 'aneurysm', 'inherited disorders', 'muscle issues like shoulder', 'shortness of breath disorientation', 'obstructive disease', 'pericarditis', 'alcohol mistreating', "Turner's", 'coronary artery calcification', 'cramps', 'varicose', 'nail', 'chest tightness', 'tingling', 'dehydration', 'premature coronary diseases familial', 'magnesium iron', 'heart disease', 'fever', 'strokes', 'vitamin D.', 'blood disorders', 'sneeze', 'vitamin', 'cardiac death', 'prediabetes', 'obstructive lesion', 'failure', 'weakness', 'diabetes', 'coronary disease friends', 'sinus infection', 'heart murmur', 'palpitations', 'genetic disorder', 'ADD', "Raynaud's", 'coronary calcification', 'diabetes', 'emphysema', 'diabetic coma', 'anxiety', 'heart failure cardiomyopathy', 'shortness of breath','general disorientation', 'dizziness', 'headache','bruising', 'loss', 'liver damage', 'chest pains', 'shock', 'homicide', 'premature coronary disease familial', 'panic', 'vascular disease', 'poisoning', 'clotting disorders', 'cancer', 'vertigo', 'panic attacks', 'kidney stones', 'appetite', 'HA', 'nicotine', 'kidney infections', 'heart attacks', 'sleep disorders', 'allergic', 'diabetes stroke', 'low carb eating', 'trauma', 'allergic reaction', 'burn', 'fainting', 'atrial septal defect', 'shortness of breath and stuff', 'hypertension', 'sweats', 'cramping', 'constipation', 'insomnia', 'vitamin C', 'muscular probable Musco skeletal pain', 'familial', 'delusions', 'bad reflux', 'hysterectomy', 'death', 'GI tract reflux', 'obstructive coronary disease', 'exertional', 'prolapse prolapse', 'sick anxiety', 'cholesterol disorder', 'thyroid disorders', 'calcification', 'headaches', 'apnea', 'herpes', 'shortness of breath nothing', 'sneezing', 'fatty liver', 'colon cancer']
        # Dynamic list of diseases
        pat_d_f_list = final_diseases_list

    #making a list of organs
        organs_list=['Mouth', 'head','lower back', 'my back', 'leg', 'Teeth', 'Tongue', 'Salivary glands', 'Parotid glands', 'Submandibular glands', 'Sublingual glands', 'Larynx', 'Esophagus', 'Stomach', 'Small intestine', 'Duodenum', 'Jejunum', 'Ileum', 'Large intestine', 'Liver', 'Gallbladder', 'Mesentery', 'Pancreas', 'Anal canal and anus', 'Blood cells', 'Respiratory system', 'Nasal cavity', 'Pharynx', 'Larynx', 'Trachea', 'Bronchi', 'Lungs', 'Diaphragm', 'Urinary system', 'Kidneys', 'Ureter', 'Bladder', 'Urethra', 'Reproductive organs', 'Female reproductive system', 'Internal reproductive organs', 'Ovaries', 'Fallopian tubes', 'Uterus', 'Vagina', 'External reproductive organs', 'Vulva', 'Clitoris', 'Placenta', 'Male reproductive system', 'Internal reproductive organs', 'Testes', 'Epididymis', 'Vas deferens', 'Seminal vesicles', 'Prostate', 'Bulbourethral glands', 'External reproductive organs', 'Penis', 'Scrotum', 'Endocrine system', 'Pituitary gland', 'Pineal gland', 'Thyroid gland', 'Parathyroid glands', 'Adrenal glands', 'Pancreas', 'Circulatory system', 'Circulatory system', 'Heart', 'Patent Foramen Ovale', 'Arteries', 'Veins', 'Capillaries', 'Lymphatic system', 'Lymphatic vessel', 'Lymph node', 'Bone marrow', 'Thymus', 'Spleen', 'Gut-associated lymphoid tissue', 'Tonsils', 'Interstitium', 'Nervous system', 'Brain', 'Cerebrum', 'Cerebral hemispheres', 'Diencephalon', 'The brainstem', 'Midbrain', 'Pons', 'Medulla oblongata', 'Cerebellum', 'The spinal cord', 'The ventricular system', 'Choroid plexus', 'Peripheral nervous system', 'Nerves', 'Cranial nerves', 'Spinal nerves', 'Ganglia', 'Enteric nervous system', 'Sensory organs', 'Eye', 'Cornea', 'Iris', 'Ciliary body', 'Lens', 'Retina', 'Ear', 'Outer ear', 'Earlobe', 'Eardrum', 'Middle ear', 'Ossicles', 'Inner ear', 'Cochlea', 'Vestibule of the ear', 'Semicircular canals', 'Olfactory epithelium', 'Tongue', 'Taste buds', 'Integumentary system', 'Mammary glands', 'Skin', 'Subcutaneous tissue']

    #extracting complaints = sentences containing diseases + sentences containing organ mentions
        disease_complaints=[]
        organ_complaints=[]
        patient_complaints = []     #disease_complaints + organ_complaints
        
        patient_diseases = []
        mentioned_organs=[]
        SOAP=[]
        for line in pat_text.split('.'):
            line= str(line)+"."
            for i in pat_d_f_list:
                if " "+i+" " in line or " "+i+"." in line or  (line[0]==i[0] and i+" " in line):
                    if line not in disease_complaints:
                        disease_complaints.append(line)
                    if line not in patient_complaints:
                        patient_complaints.append(line)
                    patient_diseases.append(i)
            for i in organs_list:
                i=i.lower()
                if " "+i+" " in line.lower() or " "+i+"." in line.lower() or  (line[0].lower==i[0] and i+" " in line.lower()):
                    mentioned_organs.append(i)
                    if line not in organ_complaints:
                        organ_complaints.append(line)
                    if line not in patient_complaints:
                        patient_complaints.append(line)

                
    #removing repeated mentions
        complaints = list(set(patient_complaints))
    #converting complaints to lower case so that Capital initials won't cause problem
        for idx, line in enumerate(complaints):
            complaints[idx] = line.lower()
        
        patient_diseases=(list(set(patient_diseases)))
        mentioned_organs=(list(set(mentioned_organs)))
        disease_complaints = list(set(disease_complaints))
        organ_complaints = list(set(organ_complaints))



    #get patient family history

    #list of possessive pronouns
        pronouns_list=['whom','whose','her','my','your','his','its','hers','mine','their','our','ours','theirs']
    #list of relational pronouns
        relation_list=['ancestors', 'mom','aunt', 'child', 'uncle', 'father', 'niece', 'nephew', 'baby', 'bachelor', 'boy', 'wife', 'husband', 'partner', 'bride', 'bridegroom', 'brother', 'brother-in-law', 'child', 'cousin', 'daughter', 'daughter-in-law', 'descendants', 'elder brother', 'elder sister', 'family', 'father-in-law', 'female', 'foster-brother', 'foster-child', 'foster-father', 'foster-mother', 'foster-sister', 'girl', 'grand-child', 'grand-children', 'grand-daughter', 'grand-father', 'grand-mother', 'grand-son', 'great grand son', 'infant', 'lad', 'lover', 'maid', 'master', 'mother', 'parents', 'parent', 'sister', 'sister-in-law', 'son', 'son-in-law', 'spouse', 'step brother', 'step daughter', 'step father', 'step mother', 'step sister', 'younger brother', 'younger sister', 'younger daughter', 'younger son', 'elder daughter', 'elder son']


        
    #extracting patient family history
        disease_history=[]
        diseases_in_family=[]

        for line in complaints:
            tokns=re.split('; | |, |\*|\n',line)
            for idx,i in enumerate(tokns):
                if i in pronouns_list and (tokns[idx+1] in relation_list  or tokns[idx+1][:-1] in relation_list):
                    reln=""
                    z=2
                    reln=reln+" "+str(i)+" "+str(tokns[idx+1])
                    while (tokns[idx+z] == "'s"):
                        reln=reln+tokns[idx+z]+" "+tokns[idx+z+1]
                        z=z+1
                    for j in pat_d_f_list:
                        if j in line:
                            disease_history.append(line)
                            diseases_in_family.append(j)
                            mm=str("1. Patient's "+str(reln[3:])+" had "+str(j))
                            tmp=[]
                            tmp.append(mm)
                            SOAP.append(tmp)
                            

        # Removing patient history from patient complaints

        for i in disease_history:
            if i in disease_complaints:
                disease_complaints.remove(i)
            if i in organ_complaints:
                organ_complaints.remove(i)

        all_lines= ""
        for line in disease_complaints:
            all_lines= all_lines+" "+line

        for i in patient_diseases:
            if i not in all_lines:
                patient_diseases.remove(i)
            
        all_lines= ""
        for line in organ_complaints:
            all_lines= all_lines+" "+line
        
        for i in mentioned_organs:
            if i not in all_lines:
                mentioned_organs.remove(i)
            
            
    #extracting disease / organ enquiries by doctor

        yes_no_list=['yep', 'Yes', 'nope', 'Yeah', 'I do',  'i do have', 'yo', 'yes', 'yeah', 'no', 'Yep', 'do not', 'nop', 'nah', "don't"]
        no_list=['nope','no','do not', 'nop', 'nah', "don't"]
        yes_list=['yep', 'Yes',  'Yeah', 'I do',  'i do have', 'yo', 'yes', 'yeah',  'Yep', ]

        orgns=[]
        diseasess=[]
        disease_q_a=[]
        orgns_q_a=[]
        for idx,i in enumerate(data['clean_text']):
        
        #checking if doctor mentions any diseases
            if (data['speaker'][idx]=="Doctor"):
                for j in final_diseases_list:
                    if " "+j+" " in i or " "+j+"." in i or j+" " in i:
                        q_a_disease=[{data['speaker'][idx]:data['clean_text'][idx]}]
                    #checking if the patient response contains yes/no words
                        for yn in yes_list:
                            if " "+yn+" " in data['clean_text'][idx+1] or " "+yn+"." in data['clean_text'][idx+1] or (data['clean_text'][idx+1][0]==yn[0] and yn+" " in data['clean_text'][idx+1]):
                                if j not in diseasess:
                                    diseasess.append(j)
                                q_a_disease.append([{data['speaker'][idx+1]:data['clean_text'][idx+1]}])
                                if (q_a_disease) not in disease_q_a:
                                    disease_q_a.append(q_a_disease)
                            
        #checking if doctor mentions any organs
            if (data['speaker'][idx]=="Doctor"):
                for j in organs_list:
                    if " "+j+" " in i.lower() or " "+j+"." in i.lower() or j+" " in i.lower():
                        q_a_organ=[{data['speaker'][idx]:data['clean_text'][idx]}]
                    
                    #checking if the patient response contains yes/no words
                        for yn in yes_list:
                            if " "+yn+" " in data['clean_text'][idx+1] or " "+yn+"." in data['clean_text'][idx+1] or (data['clean_text'][idx+1][0]==yn[0] and yn+" " in data['clean_text'][idx+1]):
                                if j not in orgns:
                                    orgns.append(j)
                                q_a_organ.append([{data['speaker'][idx+1]:data['clean_text'][idx+1]}])
                                if (q_a_organ) not in orgns_q_a:
                                    orgns_q_a.append(q_a_organ)
                                break
                    
        diseasess=list(set(diseasess))
        orgns=list(set(orgns))



    # Extracting food mentions

        food_list=['Afinitor','Carmustine','Avastin','spinach','berries','brocolli','ecospirin','plavix','mediterranean diet','avocado nuts','water', 'chickens', 'ingredients', 'oats', 'rice quinoa', 'hydrochloric thiazide', 'glucose', 'Motrin', 'tobacco products', 'product', 'potassium', 'mutations', 'mutation', 'lipid', 'breakfast', 'triglyceride', 'proteins', 'aspirin', 'channel', 'meats', 'fruits', 'hemoglobin', 'triglycerides', 'vegan', 'magnesium', 'caffeine', 'tobacco', 'platelets', 'eggs', 'CRP I', 'based products', 'hormone', 'rice', 'fatty food', 'non-veg', 'magnesium vitamin', 'proteins carbohydrates', 'wheat', 'animal protein', 'plant', 'pharmacies', 'radish', 'thiazide diuretic', 'animals products', 'flaxseeds', 'vitamin C', 'unsaturated', 'LDL', 'quinoa', 'fish', 'nutrients', 'substance', 'hydrochloric', 'steroids', 'acid', 'alcohol', 'meclizine', 'enzymes', 'vitamin', 'honey', 'salt', 'CRP', 'virus', 'protein', 'wine', 'diuretic', 'products', 'iron', 'coffee', 'calcium', 'drug', 'fruit', 'animal', 'aspirin I', 'vegetable', 'animals', 'magnesium I', 'vitamin D.', 'meat products', 'vegies', 'penicillin', 'nutrient', 'animal products', 'inflammatory', 'fatty', 'multivitamin', 'magnesium iron', 'salts', 'chicken', 'antibiotics', 'proteins plant', 'hormones', 'snacks', 'lunch', 'amino', 'pea', 'foods', 'atorvastatin', 'meat', 'allergic', 'pro inflammatory type', 'calcium channel', 'carbohydrates', 'lime', 'carbon', 'meals', 'dinner', 'non-vegetarian', 'vitamins', 'vegetables', 'drugs', 'platelet', 'thiazide', 'insulin', 'Lupron']
        food_rel_words=['eat','eating','diet','intake','consume','consuming','dieting','foods','eatables',]
        yes_no_list=['yep', 'Yes', 'nope', 'Yeah', 'I do', 'okay', 'i do have', 'yo', 'yes', 'yeah', 'ok', 'no', 'Yep', 'do not', 'nop', 'nah', "don't"]
        no_list=['nope','no','do not', 'nop', 'nah', "don't"]
        yes_list=['yep', 'Yes',  'Yeah', 'I do', 'okay', 'i do have', 'yo', 'yes', 'yeah', 'ok',  'Yep', ]

        food_mentioned=[] 
        food_sentences1=[]    #using food_rel_words list    
        food_sentences2=[]  #using foods list

    # extracting food sentences using food related words
        '''for idx,i in enumerate(data['clean_text']):
            i=i.lower()
            for x in food_rel_words:
                x=x.lower()
                if ((" "+x+" ").lower() in i ) or ((" "+x+".").lower() in i ):
                    food_sentences1.append(i)
                if ((x+" ") in i and i and x[0]==i[0] ):
                    food_sentences1.append(i)
        '''
    # extracting food sentences using food list
        for i in range(len(data['clean_text'])):
            data['clean_text'][i]=data['clean_text'][i].lower()
        #print(data['clean_text'][i],data['clean_text'][i][0])
            for x in food_list:
                a=x.lower()[0]
                if ((" "+x+" ").lower() in data['clean_text'][i]) or ((" "+x+".").lower() in data['clean_text'][i]) or ((x+" ").lower() in data['clean_text'][i] and data['clean_text'][i][0]==a ):
                    if x not in food_mentioned:
                        food_mentioned.append(x)
                    food_sentences2.append(data['clean_text'][i])

    #check foods items in sentences in food_sentences1
        for i in food_list:
            if " "+i+" ".lower() in food_sentences1 and " "+i+" ".lower() not in food_mentioned:
                food_mentioned.append(i)
            

    #final food sentences list, combining food_sentences1 and food_sentences2
        food_sentences=food_sentences1
        [food_sentences.append(x) for x in food_sentences2 if x not in food_sentences] 

        food_sentences=list(set(food_sentences))

    # Extracting food responses
        food_response=[]    #food responses 
        for idx,i in enumerate(data['clean_text']):
            i=i.lower()
            for x in food_sentences:
                if i==x:
                    for yn in yes_no_list:
                        if (" "+yn+" ".lower() in data['clean_text'][idx+1].lower()) or (yn+" ".lower() in data['clean_text'][idx+1].lower()) or (" "+yn+".".lower() in data['clean_text'][idx+1].lower()):
                            resp=  [ { data['speaker'][idx]:data['clean_text'][idx], data['speaker'][idx+1]:data['clean_text'][idx+1] }]
                            if resp not in food_response:
                                food_response.append(resp)


                            
        patient_diseases=list(reversed(sorted(patient_diseases,key=len)))
        updated_list=[]
        for k in range(len(disease_complaints)):
        #print(disease_complaints[k])
            tmp=disease_complaints[k]
        #print('tmp',tmp)
            for i in patient_diseases:   
                if i in tmp:
                #print(i)
                    updated_list.append(i)
                    tmp=tmp.replace(i,'')
                
        #print(i,tmp)
    #updated_list


        diseases_in_family=list(reversed(sorted(diseases_in_family,key=len)))
        updated_list1=[]
        for k in range(len(disease_history)):
        #print(disease_complaints[k])
            tmp=disease_history[k]
        #print('tmp',tmp)
            for i in diseases_in_family:   
                if i in tmp:
                #print(i)
                    updated_list1.append(i)
                    tmp=tmp.replace(i,'')
                
        #print(i,tmp)
        updated_list1=list(set(updated_list1))
                
        disease_history=list(set(disease_history))
    #print('*___LIST OF diseases FROM PATIENT___*')
    #print(updated_list)
        '''
        summarized_dis=[]
    #print('\n*___LIST OF DISEASE COMPLAINTS FROM PATIENT___*')
        for i in range(len(disease_complaints)):
            text=str(disease_complaints[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=20,
                                        max_length=40,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_dis.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_dis.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
    #print('*___LIST OF organs FROM PATIENT___*')
    #print(mentioned_organs)
        summarized_orgs=[]
        for i in range(len(organ_complaints)):
            text=str(organ_complaints[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n", text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=20,
                                        max_length=40,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_orgs.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_orgs.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
        print("_______________________________________________________")
        '''
        '''
        
        summarized_hist=[]
        for i in range(len(disease_history)):
            text=str(disease_history[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_hist.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_hist.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)


        s=""
        summary_food=[]
        for i in range(len(food_sentences)):
            tex=str(food_sentences[i])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',tex)
            if(len(k)>70):
                t5_prepared_Text = "summarize: "+tex
            #print ("original text preprocessed: \n",tex)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=50,
                                        max_length=150,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summary_food.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summary_food.append(tex)
            #print('\n','"This has no summary"---',tex,'\n')

        summarized_disease=[]
        summarized_disease1=[]

        for i in range(len(disease_q_a)):
            text=str(disease_q_a[i][0]['Doctor'])
            text1=str(disease_q_a[i][1][0]['Patient'])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            k1=re.split(r'\s+',text1)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_disease.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_disease.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
            if(len(k1)>50):
                t5_prepared_Text = "summarize: "+text1
                print ("original text preprocessed: \n",text1)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_disease1.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_disease1.append(text1)
            #print('"This has no summary"',text1,'\n')
        #print(idx,i)

    #print("_______________________________________________________")


    #print("\n\n\nDiseases list:\n")
    #print(diseasess)

        for i in range(len(disease_q_a)):
            for j in diseasess:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in disease_q_a[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    summarized_disease[i]=summarized_disease[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_disease1[i]=summarized_disease1[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_disease[i]=summarized_disease[i].replace(" "+str(j)+"."," "+hlt+".")
                    summarized_disease1[i]=summarized_disease1[i].replace(" "+str(j)+"."," "+hlt+".")
        for i in range(len(disease_q_a)):
            for j in diseasess:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in disease_q_a[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    disease_q_a[i][0]['Doctor']=disease_q_a[i][0]['Doctor'].replace(" "+str(j)+" "," "+hlt+" ")
                    disease_q_a[i][0]['Patient']=disease_q_a[i][1][0]['Patient'].replace(" "+str(j)+" "," "+hlt+" ")
                    disease_q_a[i][0]['Doctor']=disease_q_a[i][0]['Doctor'].replace(" "+str(j)+"."," "+hlt+".")
                    disease_q_a[i][0]['Patient']=disease_q_a[i][1][0]['Patient'].replace(" "+str(j)+"."," "+hlt+".")
                
        

        summarized_orgdis=[]
        summarized_orgdis1=[]
        for i in range(len(orgns_q_a)):
            text=str(orgns_q_a[i][0]['Doctor'])
            text1=str(orgns_q_a[i][1][0]['Patient'])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            k1=re.split(r'\s+',text1)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
                print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_orgdis.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_orgdis.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
            if(len(k1)>50):
                t5_prepared_Text = "summarize: "+text1
            #print ("original text preprocessed: \n",text1)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_orgdis1.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_orgdis1.append(text1)
            #print('"This has no summary"',text1,'\n')
        #print(idx,i)

    #print("\n\nOrgans list:\n")
    #print(orgns)

        
    for i in range(len(orgns_q_a)):
        for j in orgns:
            if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in orgns_q_a[i]:
                hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                orgns_q_a[i][0]['Doctor']=orgns_q_a[i][0]['Doctor'].replace(" "+str(j)+" "," "+hlt+" ")
                orgns_q_a[i][0]['Patient']=orgns_q_a[i][1][0]['Patient'].replace(" "+str(j)+" "," "+hlt+" ")
                orgns_q_a[i][0]['Doctor']=orgns_q_a[i][0]['Doctor'].replace(" "+str(j)+"."," "+hlt+".")
                orgns_q_a[i][0]['Patient']=orgns_q_a[i][1][0]['Patient'].replace(" "+str(j)+"."," "+hlt+".")

        
        for i in range(len(orgns_q_a)):
            for j in orgns:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in orgns_q_a[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    summarized_orgdis[i]=summarized_orgdis[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_orgdis1[i]=summarized_orgdis1[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_orgdis[i]=summarized_orgdis[i].replace(" "+str(j)+"."," "+hlt+".")
                    summarized_orgdis1[i]=summarized_orgdis1[i].replace(" "+str(j)+"."," "+hlt+".")            
                
                

        summarized_foodr=[]
        summarized_foodr1=[]

        for i in range(len(food_response)):
            text=str(food_response[i][0]['Doctor'])
            text1=str(food_response[i][0]['Patient'])
        #preprocess_text = text.strip().replace("\n","")
            k=re.split(r'\s+',text)
            k1=re.split(r'\s+',text1)
            if(len(k)>50):
                t5_prepared_Text = "summarize: "+text
            #print ("original text preprocessed: \n",text)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_foodr.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_foodr.append(text)
            #print('"This has no summary"',text,'\n')
        #print(idx,i)
            if(len(k1)>50):
                t5_prepared_Text = "summarize: "+text1
            #print ("original text preprocessed: \n",text1)

                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
                summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=50,
                                        early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            #summarized_foodr1.append(output)
            #print ("\nSummarized text: \n",output,'\n\n')
            else:
                summarized_foodr1.append(text1)
            #print('"This has no summary"',text1,'\n')
        #print(idx,i)

        
    for i in range(len(food_response)):
        for j in food_list:
            if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in food_response[i]:
                hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                food_response[i][0]['Doctor']=food_response[i][0]['Doctor'].replace(" "+str(j)+" "," "+hlt+" ")
                food_response[i][0]['Patient']=food_response[i][0]['Patient'].replace(" "+str(j)+" "," "+hlt+" ")
                food_response[i][0]['Doctor']=food_response[i][0]['Doctor'].replace(" "+str(j)+"."," "+hlt+".")
                food_response[i][0]['Patient']=food_response[i][0]['Patient'].replace(" "+str(j)+"."," "+hlt+".")

        for i in range(len(food_response)):
            for j in food_list:
                if ' '+str(j).lower()+'.' or ' '+str(j).lower()+' 'or ''+str(j).lower()+' ' in food_mentioned[i]:
                    hlt='\033[94m'+'\033[1m'+'\033[4m'+str(j).upper()+'\033[4m'+'\033[0m'+'\033[91m'+'\033[0m'
                    summarized_foodr[i]=summarized_foodr[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_foodr1[i]=summarized_foodr1[i].replace(" "+str(j)+" "," "+hlt+" ")
                    summarized_foodr[i]=summarized_foodr[i].replace(" "+str(j)+"."," "+hlt+".")
                    summarized_foodr1[i]=summarized_foodr1[i].replace(" "+str(j)+"."," "+hlt+".")            
        '''
                
        #print('KEYWORDS PRESENT IN WHOLE TRANSCRIPTION\n')
        transcription=[]
        disease_complaint=[]
        for i in disease_complaints:
            if i.lower() not in disease_history:
                disease_complaint.append(i)
        for i in range(len(disease_complaint)):
            disease_complaint[i]=str(i+1)+'.'+disease_complaint[i]       
        #print('*___LIST OF DISEASES FROM PATIENT___*')
        #print(*updated_list,sep='\n')
        transcription.append(updated_list)
        SOAP.append(disease_complaint)
                                                
        s=''
        for i in updated_list:
            s+=str(i)+', '                                        
        mm=str('This consultation is regarding the following points:\n' +str(s[:-2]))
        tmp=[]
        tmp.append(mm)
        SOAP.append(tmp)
        #transcription.append(disease_complaint)
        #print('\n*___LIST OF ORGANS FROM PATIENT___*')
        #transcription.append(mentioned_organs)
        #transcription.append(organ_complaints)

        #print(*mentioned_organs,sep='\n')
        #print("\n*___LIST OF DISEASES IN FAMILY___*")
        #print(*updated_list1,sep='\n')
        #transcription.append(updated_list1)
        #transcription.append(disease_history)
        #print("\n*___LIST OF FOODs MENTIONED___*")
        #transcription.append(food_mentioned)
        #print(*food_mentioned,sep='\n')
        s=''
        for i in food_mentioned:
            s+=str(i)+', '
        mm=str('Patient has been recommended to take '+str(s[:-2]))
        tmp=[]
        tmp.append(mm)
        SOAP.append(tmp)
        
        #transcription.append(food_sentences)
        
        test=['test','report','tests','reports']
        terms=['angiography','maze surgery','CT scan','MRI']
        k=[]
        for i in test:
            if i in doc_text:
                for j in terms:
                    if j in doc_text:
                        k.append(j)

        transcription.append(list(set(k)))
        k=list(set(k))
        s=''
        for i in k:
            s+=str(i)+', '
        mm=str('Patient has undergone '+str(s[:-2]))
        tmp=[]
        tmp.append(mm)
        SOAP.append(tmp)

        '''print('NOTE\n')
    #highlighting the normal response

        disease_complaints = make_bold(updated_list,disease_complaints)
        print('\n*___LIST OF DISEASE COMPLAINTS FROM PATIENT___*')


    #highlighting the summarized response
        summarized_dis= make_bold(updated_list,summarized_dis)


        for i in range(len(disease_complaints)):
            print(summarized_dis[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')




        organ_complaints = make_bold(mentioned_organs,organ_complaints)


    #highlighting the summarized response
        summarized_orgs= make_bold(mentioned_organs,summarized_orgs)

        print('\n*___LIST OF ORGAN COMPLAINTS FROM PATIENT___*')

        for i in range(len(organ_complaints)):
            print(summarized_orgs[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')
        
        


        disease_history = make_bold(updated_list1,disease_history)


        #highlighting the summarized response
        summarized_hist= make_bold(updated_list1,summarized_hist)
        print("\n\n__**DISEASES History IN FAMILY**__")

        for i in range(len(disease_history)):
            print(summarized_hist[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')




        food_sentences = make_bold(food_mentioned,food_sentences)


    #highlighting the summarized response
        summary_food= make_bold(food_mentioned,summary_food)
        print("\nFood related sentences:\n")


        for i in range(len(food_sentences)):
            print(summary_food[i],'\n')
        print('\n------------------------------------------------------------------------------------\n')


        print('\nDISCUSSIONS\n\n')

        print("\n\n*__Disease discussion between doctor and patient:")
        for i in range(len(disease_q_a)):

            print('Doctor :',summarized_disease[i],'\n')
        
            print('Patient :',summarized_disease1[i],'\n')
            print('\n---------------------------------------------------------------------------------\n')
        
        

        print("\n\n*__Organ discussion between doctor and patient:")
        for i in range(len(orgns_q_a)):
            print('Doctor :',summarized_orgdis[i],'\n')
            print('Patient :',summarized_orgdis1[i],'\n')
            print('\n---------------------------------------------------------------------------------\n')

        

        print("\n\n*__Food discussion between doctor and patient:")
        for i in range(len(food_response)):
            print('Doctor :',summarized_foodr[i],'\n')
            print('Patient :',summarized_foodr1[i],'\n')
            '''
        #print('\n---------------------------------------------------------------------------------\n')
            
        
        
        return render_template('soap.html', prediction_text1=SOAP[2] , prediction_text2=SOAP[4] ,prediction_text3=SOAP[0]+SOAP[1],prediction_text4=SOAP[3])

       



if __name__ == '__main__':
    app.run(debug=True)