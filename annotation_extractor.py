#Reports
import os
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import time
from sklearn.model_selection import train_test_split
start = time.time()
class MimicAnnotation():
    def __init__(self,report_path, image_path):
        super(MimicAnnotation, self).__init__()
        self.report_path = report_path
        self.image_path = image_path
    
    def report(self, report_path):
        dir_list = os.listdir(self.report_path)
        paths=[]
        for i in dir_list:
            #print(i)
            pa= (os.listdir(self.report_path+'/'+i))
            paths.append(pa)

        new_path=[]
        for i in range(len(dir_list)):
            for j in paths[i]:
               # print(j[0:3])
                if j[0:3]==dir_list[i]:
                    new_paths=os.path.join(self.report_path,dir_list[i],j)
                    new_path.append(new_paths)
                    #print(new_paths)
        road_to_report=[]
        for i in new_path:
            final_dest=os.listdir(i)
            for j in final_dest:
                text_paths=os.path.join(i,j)
                #print(text_paths)
                road_to_report.append(text_paths)
        impression_reports=[]
        findings_reports=[]
        rep_sub_id=[]
        for i in road_to_report:
        #print(i)
            with open(i) as f:
                report = f.read()
                if report != ' ':
                    findings_start = report.find('FINDINGS:')  # find the start index of 'FINDINGS:'
                    impression_start = report.find('IMPRESSION:')
                    if findings_start != -1 and impression_start != -1:
                        subject = os.path.splitext(os.path.basename(i))[0]
                    #rep_sub_id.append(subject) this is for find or imp
                        #impression_start != -1:
                        #subject = os.path.splitext(os.path.basename(i))[0]
                        rep_sub_id.append(subject)
                        text = report[impression_start+11:]  # extract text after 'IMPRESSION:' (assuming 11 is the length of 'IMPRESSION:')
                        #print(text)
                        end_pos = 0
                        count = 0
                        for j in range(len(text)):
                            #print(f'text[j]:{text[j]}')
                            if text[j].isupper():
                                count += 1
                                if count == 3:
                                    print('We found three consecutive capital letters!')
                                #print(f'count:{count}')
                                if count >= 5:
                                    #print(f'count:{count}')
                                    end_pos = j - 4  # set the end position to before the first of the 5 consecutive capital letters
                                    #print(end_pos)
                                    break
                            else:
                                count = 0 # reset the count if there is a non-capital letter
                        if end_pos == 0:
                            end_pos = len(text)
                        text = text[:end_pos].replace('\n', '').replace('  ','')  # truncate text if there are 5 consecutive capital letters
                        # print(f'final text:{text}')
                        impression_reports.append(text)
                    #else:   
                        #rep_sub_id.append(subject)
                        text = report[findings_start+9:]  # extract text after 'FINDINGS:' (assuming 9 is the length of 'FINDINGS:')
                        #print(text)
                        end_pos = 0
                        count = 0
                        for j in range(len(text)):
                            #print(f'text[j]:{text[j]}')
                            if text[j].isupper():
                                count += 1
                                #print(f'count:{count}')
                                if count == 3:
                                    print('We found three consecutive capital letters!')
                                    #print ('the location is: ', i)
                                    #print ('the word is: ', text)
                                if count >= 5:
                                    #print(f'count:{count}')
                                    end_pos = j - 4  # set the end position to before the first of the 5 consecutive capital letters
                                    #print(end_pos)
                                    break
                            else:
                                count = 0 # reset the count if there is a non-capital letter
                        if end_pos == 0:
                            end_pos = len(text)
                        text = text[:end_pos].replace('\n', '').replace('  ','')  # truncate text if there are 5 consecutive capital letters
                    # print(f'final text:{text}')
                        findings_reports.append(text)
                    else:
                        print('No impressions')      
                else:
                    print('blank report path: ', i)

        print('length of collected report:', len(rep_sub_id))
        print('length of collected report(obtained from list findings_reports):', len(findings_reports))
        print('length of collected report(obtained from list impression_reports):', len(impression_reports))
        dicts_for_reports=[]
        for i in range(len(rep_sub_id)):
            #print(len(final_reports))
            report_dict={'id':rep_sub_id[i], 
                         'findings': findings_reports[i],
                         'impression': impression_reports[i] }
            #print(report_dict)
            dicts_for_reports.append(report_dict)
        print('length of dicts for reports: ', len(dicts_for_reports))
        return rep_sub_id, dicts_for_reports

    def images (self, image_path, report_path):
        
        subject_ids, dicts_for_reports= self.report(report_path)
        
        dir_list_images = os.listdir(self.image_path)
        dir_list_images=[k for k in dir_list_images if not 'index.html' in k]


        paths=[]
        for i in dir_list_images:
            #print(i)
            pa= (os.listdir(self.image_path+'/'+i))
            paths.append(pa)


        new_path=[]
        for i in range(len(dir_list_images)):
            for j in paths[i]:
                if j[0:3]==dir_list_images[i]:
                    new_paths=os.path.join(self.image_path,dir_list_images[i],j)
                    new_path.append(new_paths)


        road_to_images=[]
        for i in tqdm(new_path):
            final_dest=os.listdir(i)
            #print(final_dest)
            for j in final_dest:
                image_paths=os.path.join(i,j)
                #print(text_paths)
                road_to_images.append(image_paths)
        road_to_images=[k for k in road_to_images if not 'index.html' in k]
        print('road_to_images done')

        image_paths=[]
        for i in range(len(road_to_images)):
            try:
                dests= os.listdir(road_to_images[i])
                #print(dests)
                image_paths.append(dests)
            except:
                pass
        image_paths = []

        for dir_path in road_to_images:
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_path = os.path.join(dirpath, filename)
                        image_paths.append(image_path)

        print('entering dict for images')
        dicts_for_images = []
        for id in tqdm(subject_ids):
            id_images = []
            for image_path in image_paths:
                if id in image_path:
                    id_images.append(image_path.replace("{{", "").replace("}}", ""))
            dicts_for_images.append({'id': id, 'image_path': id_images})
        print('dict for images done')

        print('entering merged dict')

        merged_dict = []
        for report in tqdm(dicts_for_reports):
            for image in dicts_for_images:
                if report['id'] == image['id']:
                    image_report_dict = {}
                    image_report_dict['id'] = image['id']
                    image_report_dict['findings'] = report['findings']
                    image_report_dict['impression'] = report['impression']
                    image_report_dict['image_path'] = image['image_path']
                    merged_dict.append(image_report_dict)
                    #break
                else:
                    continue

        print('merged_dict done')
        print('length of merged dictionary is: ', len(merged_dict))
        merged_dict = [d for d in merged_dict if d.get('impression') != ' ']
        merged_dict = [d for d in merged_dict if d.get('findings') != ' ']
        print('filtred merged_dict done')
        print('length of filtered merged dictionary is: ', len(merged_dict))
        split_csv=pd.read_csv(r"/home/debodeep.banerjee/R2Gen/data/mimic/mimic-cxr-2.0.0-split.csv")
        split_csv['study_id']=['s'+str(i) for i in split_csv['study_id']] 
        split_train=split_csv[split_csv['split']=='train']
        split_test=split_csv[split_csv['split']=='test']
        split_val=split_csv[split_csv['split']=='validate']
        print('length of split dataset; ', len(split_train)+len(split_test)+len(split_val))
        new_split_csv=split_csv[['study_id', 'split']]
        loaded_df=pd.DataFrame(merged_dict)
        loaded_df= loaded_df.rename(columns={'id':'study_id'})
        merged_data=pd.merge(new_split_csv,loaded_df, on='study_id')
        merged_data = merged_data.drop_duplicates(subset='study_id')
        merged_data = merged_data.reset_index(drop=True)
        print('printing merged df..')
        print(merged_data.head())
        merged_train_portion = merged_data[merged_data['split'] == 'train']
        train_indices = merged_train_portion.index[-(int(len(merged_train_portion)*0.3)):]
        # Convert the last 30% to 'finetune'
        merged_data.loc[train_indices, 'split'] = 'finetune'
        print('length of train+finetune portion: ', len(merged_data))#[merged_data['split']=='train']))
        print('length of training portion: ', len(merged_data[merged_data['split']=='train']))
        print('length of finetune portion: ', len(merged_data[merged_data['split']=='finetune']))
        train_df = merged_data[merged_data['split']=='train']
        fine_tune_df = merged_data[merged_data['split']=='finetune']
        #train_df, remaining_df = train_test_split(loaded_df, test_size=0.3, random_state=42)
        #train_df['split'] = 'train'
        #remaining_df['split'] = None  # Initialize the split column for remaining_df

        # Step 2: Split the remaining set into fine-tuning and remaining sets
        #fine_tune_df, remaining_df = train_test_split(remaining_df, train_size=0.8333, random_state=42)
        #fine_tune_df['split'] = 'finetune'
        #remaining_df['split'] = None  # Reset the split column for remaining_df

        # Step 3: Split the remaining set into test and validation sets
        #test_df, val_df = train_test_split(remaining_df, test_size=0.5, random_state=42)
        test_df = merged_data[merged_data['split'] == 'test']
        val_df = merged_data[merged_data['split'] == 'validate']

        # Concatenate the dataframes back together
        final_df = pd.concat([train_df, fine_tune_df, test_df, val_df], ignore_index=True)
        csv_dict = final_df.to_dict(orient='records')
        train_box=[]
        val_box=[]
        test_box=[]
        ft_box=[]
        for i in csv_dict:
            if i['split']=='train':
                train_box.append(i)
            elif i['split']=='validate':
                val_box.append(i)
            elif i['split']=='test':
                test_box.append(i)
            else:
                ft_box.append(i)
        annotation = {'train':train_box, 'val': val_box, 'test': test_box, 'finetune': ft_box}
        out_file = open(r"/home/debodeep.banerjee/R2Gen/data/mimic/imp_and_find.json", "w")
  
        json.dump(annotation, out_file)
  
        out_file.close()
        return annotation

im_path=r'/data/MimicCXR/mimic_images/'
rep_path= r'/home/debodeep.banerjee/R2Gen/data/mimic/reports'
rep=MimicAnnotation(rep_path, im_path)
ann=rep.images(im_path, rep_path)

print('done')

print(time.time()-start)