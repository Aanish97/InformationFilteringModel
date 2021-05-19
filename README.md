# Information Filtering Model
To install the python packages, run the following command in Pycharm
<br/><br/>
pip install -r requirements.txt

## Understanding the code
There are comments in the first few lines of every function describing the purpose of the function.

## Structure of data folders
The structure is as follows;

1. BM25/
2. DIRICHLET/
3. IF-ROCCHIO-MODEL/
4. TF-IDF/
5. PROBABALISTIC_MODEL/
6. EResult1.dat
7. EResult2.dat

1-5 are folders
6-7 are .dat files

Each folder contains files for the 50 training sets, each file containing scores of the respective query in the 
different documents of the training set

**EResult1.dat** contains the Information filtering, Rocchio model scores. While the **EResult2.dat** contains
the score for the BM25 baseline model

## Packages used
**from scipy import stats**

This is the only additional package used along with numpy, it is used to calculate the t-test scores, to 
reject the null hypothesis