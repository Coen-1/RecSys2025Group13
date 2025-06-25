this codebase does the following:
- Uses reviews from meta_data files to perform aspect based sentiment analysis.
- Removes the reviews from the meta_data file and adds aspect: sentiment strings to the title of the product in the meta_data file.



# pre train data
to generate pretrain data augmented with aspect based sentiment analysis, the following steps need to be followed.


## 1 get a metadata file where the reviews are present as follows:
==============================================================================================================================
Product ID: B00062Z0VQ
{
  "title": "K&amp;N 33-2767 High Performance Replacement Air Filter",
  "brand": "K&amp;N",
  "category": "Automotive Replacement Parts Filters Air Filters &amp; Accessories Air Filters",
  "review_A2QWIF40QE6HHO": "she breathes so much easier now!  my little car is a dream to drive now, i love her, this filter is pre lubed for your driving pleasure",
  "review_A2XLE3MHK1DMHT": "Fitment is good! The car winds up a little faster, nothing earth shattering. I will however, add an overdrive pulley to increase the supercharger boost. This K&N filter will be a good addition to that. This filter will take you to the 200hp flywheel club.",
  "review_A3L9F78ITLVCZU": "Fine product!",
  "review_A3N5Z2IP1SBI42": "great stuff fast ship  thanks much",
  "review_AX44CBRWEK8AC": "Replaced my air filter with this unit.  Car seems to perform well - No problems so far.",
  "review_A2CSW64QQ5I68": "Snapped into my M-B SLK 230 Compressor like a breeze."
}
==============================================================================================================================



## 2 run the following commands or use the snellius job file
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt


python main.py \
    --input pretrain_data_augmentation/data/meta_data.json \
    --output pretrain_data_augmentation/sentiments.json

This generates the following aspect based sentiment per review product combination

B004QHSJ5G_review_A2DS81LF6LEXCO": [
    {
      "aspect": "money",
      "sentiment": "Positive",
      "confidence": 0.9937
    },
    {
      "aspect": "fitment",
      "sentiment": "Positive",
      "confidence": 0.9576
    }
  ],


## 3 convert the meta_data file to include the aspects and sentimentes by running the following

cd pretrain_data_augmentation
python add_aspect_sentiment_to_meta_data.py
python filter_train_and_dev_data.py


the output will be as follows:


Product ID: B004QHSJ5G
{
  "title": "Muteki 41886L Purple 12mm x 1.5mm Closed End Spline Drive Lug Nut Set with Key, (Set of 20) | rust: Positive, bag: Negative, lug nuts: Negative, look: Positive, color: Positive, lug: Neutral, wheels: Neutral, fit: Neutral, work: Positive, paint: Negative, vhicule: Negative, lug nuts: Neutral, blue: Neutral, wheel: Positive, key: Negative, blue: Positive, paint: Negative, paint: Positive, wheels: Positive, lug nut tips: Positive, lugs: Positive, price: Positive, installing: Positive, coating: Positive, fit: Positive, rims: Positive, money: Positive, fitment: Positive, wheels: Positive, fit: Positive, shipping: Positive, price: Positive, rims: Neutral, locking spline drive: Positive",
  "brand": "MUTEKI",
  "category": "Automotive Tires & Wheels Accessories & Parts Lug Nuts & Accessories Lug Nuts"
}






# fine tune data
to generate finetune data augmented with aspect based sentiment analysis, the following steps need to be followed.


## 1 get a metadata file where the reviews are present as follows:
==============================================================================================================================
Product ID: B00062Z0VQ
{
  "title": "K&amp;N 33-2767 High Performance Replacement Air Filter",
  "brand": "K&amp;N",
  "category": "Automotive Replacement Parts Filters Air Filters &amp; Accessories Air Filters",
  "review_A2QWIF40QE6HHO": "she breathes so much easier now!  my little car is a dream to drive now, i love her, this filter is pre lubed for your driving pleasure",
  "review_A2XLE3MHK1DMHT": "Fitment is good! The car winds up a little faster, nothing earth shattering. I will however, add an overdrive pulley to increase the supercharger boost. This K&N filter will be a good addition to that. This filter will take you to the 200hp flywheel club.",
  "review_A3L9F78ITLVCZU": "Fine product!",
  "review_A3N5Z2IP1SBI42": "great stuff fast ship  thanks much",
  "review_AX44CBRWEK8AC": "Replaced my air filter with this unit.  Car seems to perform well - No problems so far.",
  "review_A2CSW64QQ5I68": "Snapped into my M-B SLK 230 Compressor like a breeze."
}
==============================================================================================================================


## 2 run the following commands or use the snellius job file
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt


python main.py \
    --input finetune_data_augmentation/data/Scientific/meta_data.json\
    --output finetune_data_augmentation/sentiments.json

This generates the following aspect based sentiment per review product combination

B004QHSJ5G_review_A2DS81LF6LEXCO": [
    {
      "aspect": "money",
      "sentiment": "Positive",
      "confidence": 0.9937
    },
    {
      "aspect": "fitment",
      "sentiment": "Positive",
      "confidence": 0.9576
    }
  ],


## 3 convert the meta_data file to include the aspects and sentimentes by running the following
cd finetune_data_augmentation
python add_aspect_sentiment_to_meta_data.py






# Instructions for running on the snellius cluster

ssh -X scur2758@snellius.surf.nl

### make the venv and get requirements
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt


### Run the job
sbatch sentiment.job
  - note the job id returned --> Submitted batch job 1234567


### job status
squeue -u scur2758

### job status specific
scontrol show job 12283361


### view output

# To view the entire output file
cat jobs/job_output/12256597.out


scancel 12493614

