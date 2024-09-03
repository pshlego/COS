detect_aggregation_query_prompt = """Using f_agg() api to detect aggregation query.

question: when was the third highest paid Rangers F.C . player born ?
The answer is : f_agg([True])

question: what is the full name of the Jesus College alumni who graduated in 1960 ?
The answer is : f_agg([False])

question: how tall , in feet , is the Basketball personality that was chosen as MVP most recently ?
The answer is : f_agg([True])

question: what is the highest best score series 7 of Ballando con le Stelle for the best dancer born 3 July 1969 ?
The answer is : f_agg([True])

question: which conquerors established the historical site in England that attracted 2,389,548 2009 tourists ?
The answer is : f_agg([False])

question: what is the NYPD Blue character of the actor who was born on January 29 , 1962 ?
The answer is : f_agg([False])

question: {question}
The answer is : 
"""

select_row_wise_prompt = """Using f_row() api to select relevant rows in the given table and linked passages that support or oppose the question.
please use f_row([*]) to select all rows in the table.

/*
table caption : list of rangers f.c. records and statistics
col : # | player | to | fee | date
row 1 : 1 | alan hutton | tottenham hotspur | 9,000,000 | 30 january 2008
row 2 : 2 | giovanni van bronckhorst | arsenal | 8,500,000 | 20 june 2001
row 3 : 3 | jean-alain boumsong | newcastle united | 8,000,000 | 1 january 2005
row 4 : 4 | carlos cuellar | aston villa | 7,800,000 | 12 august 2008
row 5 : 5 | barry ferguson | blackburn rovers | 7,500,000 | 29 august 2003
*/

/*
passages linked to row 1
- alan hutton: alan hutton ( born 30 november 1984 ) is a scottish former professional footballer , who played as a right back . hutton started his career with rangers , and won the league title in 2005 . he moved to english football with tottenham hotspur in 2008 , and helped them win the league cup later that year .
- tottenham hotspur f.c.: tottenham hotspur football club , commonly referred to as tottenham ( /ˈtɒtənəm/ ) or spurs , is an english professional football club in tottenham , london , that competes in the premier league .
passages linked to row 2
- giovanni van bronckhorst: giovanni christiaan van bronckhorst oon ( dutch pronunciation : [ ɟijoːˈvɑni vɑm ˈbrɔŋkɦɔrst ] ( listen ) ; born 5 february 1975 ) , also known by his nickname gio , is a retired dutch footballer and currently the manager of guangzhou r & f . formerly a midfielder , he moved to left back later in his career .
- arsenal f.c.: arsenal football club is a professional football club based in islington , london , england , that plays in the premier league , the top flight of english football . the club has won 13 league titles , a record 13 fa cups , 2 league cups , 15 fa community shields , 1 league centenary trophy , 1 uefa cup winners ' cup and 1 inter-cities fairs cup . 
passages linked to row 3
- jean-alain boumsong: jean-alain boumsong somkong ( born 14 december 1979 ) is a former professional football defender , including french international . he is known for his physical strength , pace and reading of the game .
- newcastle united f.c.: newcastle united football club is an english professional football club based in newcastle upon tyne , tyne and wear , that plays in the premier league , the top tier of english football . founded in 1892 by the merger of newcastle east end and newcastle west end . 
passages linked to row 4
- carlos cuéllar: carlos javier cuéllar jiménez ( spanish pronunciation : [ ˈkaɾlos ˈkweʎaɾ ] ; born 23 august 1981 ) is a spanish professional footballer who plays for israeli club bnei yehuda . mainly a central defender , he can also operate as a right back .
- aston villa: aston villa football club ( nicknamed villa ) is an english professional football club based in aston , birmingham . the club competes in the premier league , the top tier of the english football league system . founded in 1874 , they have played at their home ground , villa park , since 1897 .
*/

question : when was the third highest paid rangers f.c . player born ?
The answer is : f_row([row 3])


/*
{table}
*/

/*
{linked_passages}
*/

question: {question}
The answer is : 
"""

select_passages_prompt_v2 = """Given a question and a table segment, return a list of passage titles linked to the table segment that contain the information needed to answer the question. NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT list().


/*
table caption : List of politicians, lawyers, and civil servants educated at Jesus College, Oxford
col : Name | M | G | Degree | Notes
row 1 : Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993
*/

/*
Title: Lalith Athulathmudali. Content: Lalith William Samarasekera Athulathmudali , PC ( Sinhala : ලලිත් ඇතුලත්මුදලි ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . 
Title: Law degree. Content: A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects .
Title: Oxford Union. Content: The Oxford Union Society , commonly referred to simply as the Oxford Union , is a debating society in the city of Oxford , England , whose membership is drawn primarily from the University of Oxford . Founded in 1823 , it is one of Britain 's oldest university unions and one of the world 's most prestigious private students ' societies . The Oxford Union exists independently from the university and is separate from the Oxford University Student Union .
*/

question: What is the full name of the Jesus College alumni who graduated in 1960 ?
list of relevant passages: ["Lalith Athulathmudali"]


/*
{table_segment}
*/

/*
{linked_passages}
*/

question: {question}
list of relevant passages: """