select_passage_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|> Given a question and a table segment, return a list of passage titles linked to the table segment that contain the information needed to answer the question. NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT list().
<|start_header_id|>user<|end_header_id|>
### Question
What is the full name of the Jesus College alumni who graduated in 1960 ?

### Table
Table Title: List of politicians, lawyers, and civil servants educated at Jesus College, Oxford
Name | M | G | Degree | Notes
Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993

### Linked Passages
Title: Lalith Athulathmudali. Content: Lalith William Samarasekera Athulathmudali , PC ( Sinhala : ලලිත් ඇතුලත්මුදලි ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . 

Title: Law degree. Content: A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects .

Title: Oxford Union. Content: The Oxford Union Society , commonly referred to simply as the Oxford Union , is a debating society in the city of Oxford , England , whose membership is drawn primarily from the University of Oxford . Founded in 1823 , it is one of Britain 's oldest university unions and one of the world 's most prestigious private students ' societies . The Oxford Union exists independently from the university and is separate from the Oxford University Student Union .

### List of Relevant Passages
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
['Lalith Athulathmudali']
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
### Question
{question}

### Table
{table_segment}

### Linked Passages
{linked_passages}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
### List of Relevant Passages
"""

select_table_segment_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|> Given a question, a related table, and passages linked to each row of the table, return a list of rows that contain the information needed to answer the question. NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT list().
<|start_header_id|>user<|end_header_id|>
### Question
What is the full name of the Jesus College alumni who graduated in 1960 ?

### Table & Linked Passages
Table Name: List of politicians, lawyers, and civil servants educated at Jesus College, Oxford
Column Name: Name | M | G | Degree | Notes

Row_1: Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993
Passages linked to Row 1:
- Lalith William Samarasekera Athulathmudali , PC ( Sinhala : ලලිත් ඇතුලත්මුදලි ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . 
- A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects .


Row_2: Neal Blewett ( HF ) | 1957 | 1959 | BA PPE ( 2nd ) | Member of the Australian House of Representatives ( 1977-1994 ) , Government Minister ( 1983-1994 ) , High Commissioner to the UK ( 1994-1998 )


Row_3: Joseph Clearihue | 1911 | 1914 | BA Jurisprudence ( 2nd , 1913 ) , BCL ( 3rd , 1914 ) | Canadian Rhodes scholar ; later became a member of the Legislative Assembly of British Columbia and a county court judge ; also chairman of the council of Victoria College , British Columbia ( which became the University of Victoria under his leadership )
Passages linked to Row 3:
- Joseph Badenoch Clearihue ( 1887-1976 ) was a Canadian lawyer , judge , academic and politician .
- The University of Victoria ( UVic ) is a public comprehensive research university located in the Greater Victoria municipalities of Oak Bay and Saanich , British Columbia , Canada . The university traces its roots to Victoria College , an institution established in 1903 .


### List of Relevant Rows
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
['Row_1']
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
### Question
{question}

### Table & Linked Passages
{table_and_linked_passages}

### List of Relevant Rows
"""

select_nodes_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|> Given a question, a related table, and passages linked to each row of the table, return a dictionary of rows and passages that contain the information needed to answer the question. NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
<|start_header_id|>user<|end_header_id|>
### Question
What is the full name of the Jesus College alumni who graduated in 1960 ?

### Table & Linked Passages
Table Name: List of politicians, lawyers, and civil servants educated at Jesus College, Oxford
Column Name: Name | M | G | Degree | Notes

Row_1: Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993
Passages linked to Row 1:
- Lalith Athulathmudali: Lalith William Samarasekera Athulathmudali , PC ( Sinhala : ලලිත් ඇතුලත්මුදලි ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . 
- A law degree: A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects .


Row_2: Neal Blewett ( HF ) | 1957 | 1959 | BA PPE ( 2nd ) | Member of the Australian House of Representatives ( 1977-1994 ) , Government Minister ( 1983-1994 ) , High Commissioner to the UK ( 1994-1998 )


Row_3: Joseph Clearihue | 1911 | 1914 | BA Jurisprudence ( 2nd , 1913 ) , BCL ( 3rd , 1914 ) | Canadian Rhodes scholar ; later became a member of the Legislative Assembly of British Columbia and a county court judge ; also chairman of the council of Victoria College , British Columbia ( which became the University of Victoria under his leadership )
Passages linked to Row 3:
- Joseph Clearihue: Joseph Badenoch Clearihue ( 1887-1976 ) was a Canadian lawyer , judge , academic and politician .
- The University of Victoria: The University of Victoria ( UVic ) is a public comprehensive research university located in the Greater Victoria municipalities of Oak Bay and Saanich , British Columbia , Canada . The university traces its roots to Victoria College , an institution established in 1903 .


### List of Relevant Rows
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{'rows':['Row_1'], 'passages':['Lalith Athulathmudali']}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
### Question
{question}

### Table & Linked Passages
{table_and_linked_passages}

### List of Relevant Rows
"""