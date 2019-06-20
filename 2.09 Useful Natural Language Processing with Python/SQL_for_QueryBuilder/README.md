# SQL Scripts and Query Builder Instructions

This chapter uses real de-identified clinical note examples queried from the MIMIC-III dataset. As such, you will need to obtain your own Physionet account and access to use the MIMIC dataset first. Please follow the instructions here to obtain dataset access: https://mimic.physionet.org/gettingstarted/access/

However, you will not actually need to setup the MIMIC SQL dataset locally to download the datasets required for this chapter. For each section, the necessary SQL code to query the practice datasets will be given to you to query the datasets yourself via MIMIC’s online Query Builder application: https://querybuilder-lcp.mit.edu

### Instructions:
1. For each example exercise, please go to the *query.txt file for that example. e.g. "part_a_query.txt" is for example part A.
2. Log into your Query Builder account at: https://querybuilder-lcp.mit.edu
3. Paste the SQL code into the query console.
4. Click "execute query" and wait patiently for the query to complete. This should not take more than 5 minutes.
5. Click "export result", name and save the table exactly as instructed in the book chapter or notebook (e.g. "part_a.csv").
6. Make sure you save the .csv files in the same folder as where you are running the notebook (unless you know how to redirect filepaths in the notebook).
7. The sql queries are meant to work directly on Query Builder. If you are planning to query directly from the notebook by connecting to a local mysql or postgres database, then you may need to change the query so that you point to the relevant mimiciii.table everywhere you have FROM or JOIN tables.
