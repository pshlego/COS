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
The answer is : f_row([row 3])"""

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

select_passages_prompt = """Using f_passage() api to select passage titles linked to the table segment that contain the information needed to answer the question.

/*
table segment = {
  "table_caption": "list of politicians, lawyers, and civil servants educated at jesus college, oxford",
  "table_column_priority": [
    ["name", "lalith athulathmudali"],
    ["m", "1955"],
    ["g", "1960"],
    ["degree", "ba jurisprudence ( 2nd , 1958 ) , bcl ( 2nd , 1960 )"],
    ["notes", "president of the oxford union ( 1958 ) ; a sri lankan politician ; killed by the tamil tigers in 1993"]
  ]
}
*/
/*
linked passages = {
  "linked_passage_title": [
    "Lalith Athulathmudali",
    "Law degree",
    "Oxford Union",
    "Liberation Tigers of Tamil Eelam"
  ],
  "linked_passage_context": [
    ["Lalith Athulathmudali", "Lalith William Samarasekera Athulathmudali , PC ( Sinhala : \u0dbd\u0dbd\u0dd2\u0dad\u0dca \u0d87\u0dad\u0dd4\u0dbd\u0dad\u0dca\u0db8\u0dd4\u0daf\u0dbd\u0dd2 ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . Following a failed impeachment of President Premadasa , he was removed from the UNP and formed his own party ."],
    ["Law degree", "A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects- e.g. , in Britain the Legal Practice Course is required to become a British solicitor or the Bar Professional Training Course ( BPTC ) to become a barrister ."],
    ["Oxford Union", "The Oxford Union Society , commonly referred to simply as the Oxford Union , is a debating society in the city of Oxford , England , whose membership is drawn primarily from the University of Oxford . Founded in 1823 , it is one of Britain 's oldest university unions and one of the world 's most prestigious private students ' societies . The Oxford Union exists independently from the university and is separate from the Oxford University Student Union . The Oxford Union has a tradition of hosting some of the world 's most prominent individuals across politics , academia and popular culture , including US Presidents Ronald Reagan , Jimmy Carter , Richard Nixon and Bill Clinton , British Prime Ministers Winston Churchill , Margaret Thatcher , David Cameron and Theresa May , Pakistani Prime Minister Imran Khan , activists Malcolm X , Dalai Lama and Mother Teresa , actor Morgan Freeman , musicians Sir Elton John and Michael Jackson and sportspeople Diego Maradona and Manny Pacquiao ."],
    ["Liberation Tigers of Tamil Eelam", "The Liberation Tigers of Tamil Eelam ( Tamil : \u0ba4\u0bae\u0bbf\u0bb4\u0bc0\u0bb4 \u0bb5\u0bbf\u0b9f\u0bc1\u0ba4\u0bb2\u0bc8\u0baa\u0bcd \u0baa\u0bc1\u0bb2\u0bbf\u0b95\u0bb3\u0bcd , romanized : Tami\u1e3b\u012b\u1e3ba vi\u1e6dutalaip pulika\u1e37 , Sinhala : \u0daf\u0dd9\u0db8\u0dc5 \u0d8a\u0dc5\u0dcf\u0db8\u0dca \u0dc0\u0dd2\u0db8\u0dd4\u0d9a\u0dca\u0dad\u0dd2 \u0d9a\u0ddc\u0da7\u0dd2 , romanized : Dema\u1e37a \u012b\u1e37\u0101m vimukti ko\u1e6di , commonly known as the LTTE or the Tamil Tigers ) was a Tamil extreme rightwing militant organisation that was based in northeastern Sri Lanka . Its aim was to secure an independent state of Tamil Eelam in the north and east . Founded in May 1976 by Velupillai Prabhakaran , with the support of Indian military research and analyse wing it was involved in armed clashes against the Sri Lankan state forces and by the late 1980s was the dominant Tamil militant group in Sri Lanka . The escalation of intermittent conflict into a full-scale nationalist insurgency however did not commence before the countrywide pogroms against Tamils . Since 1983 , more than 80,000 have been killed in the civil war that lasted 26 years . The LTTE which started out as a guerrilla force , over time , increasingly came to resemble that of a conventional fighting force with a well-developed military wing that included a navy , an airborne unit , an intelligence wing , and a specialised suicide attack unit . It is designated as a terrorist organisation by 32 countries , including the European Union , Canada , the United States , and India . The Indian state 's relationship with the LTTE in particular , was complex , as it went from initially supporting the organisation to engaging it in direct combat through the Indian Peace Keeping Force , owing to changes in the former 's foreign policy during the phase of the conflict . It was known for using women and children in combat and is recognised for having carried out a number of high-profile assassinations , including Sri Lankan President Ranasinghe Premadasa in 1993 and former Indian Prime Minister Rajiv Gandhi in 1991 . Over the course of the conflict , the Tamil Tigers frequently exchanged control of territory in the north-east with the Sri Lankan military , with the two sides engaging in intense military confrontations . It was involved in four unsuccessful rounds of peace talks with the Sri Lankan government and at its peak in 2000 , the LTTE was in control of 76% of the landmass in the Northern and Eastern provinces of Sri Lanka . Prabhakaran headed the organisation from its inception until his death in 2009 ."]
  ]
}
*/
question : What is the full name of the Jesus College alumni who graduated in 1960 ?.
The answer is : f_passage([Lalith Athulathmudali])

/*
table segment = {
  "table_caption": "list of rangers f.c. records and statistics",
  "table_column_priority": [
    ["#", "3"],
    ["player", "jean-alain boumsong"],
    ["to", "newcastle united"],
    ["fee", "\u00a38,000,000"],
    ["date", "1 january 2005"]
  ]
}
*/
/*
linked passages = {
  "linked_passage_title": [
    "Jean-Alain Boumsong",
    "Newcastle United F.C."
  ],
  "linked_passage_context": [
    ["Jean-Alain Boumsong", "Jean-Alain Boumsong Somkong ( born 14 December 1979 ) is a former professional football defender , including French international . He is known for his physical strength , pace and reading of the game ."],
    ["Newcastle United F.C.", "Newcastle United Football Club is an English professional football club based in Newcastle upon Tyne , Tyne and Wear , that plays in the Premier League , the top tier of English football . Founded in 1892 by the merger of Newcastle East End and Newcastle West End . Per the requirements from the Taylor Report , in response to the Hillsborough disaster , that all Premiership teams have an all-seater stadium , the grounds were adjusted in the mid-1990s and now has a capacity of 52,354. , The team plays its home matches at St James ' Park . The club has been a member of the Premier League for all but three years of the competition 's history , spending 86 seasons in the top tier as of May 2018 , and has never dropped below English football 's second tier since joining the Football League in 1893 . They have won four League Championship titles , six FA Cups and a Charity Shield , as well as the 1969 Inter-Cities Fairs Cup and the 2006 UEFA Intertoto Cup , the ninth highest total of trophies won by an English club . The club 's most successful period was between 1904 and 1910 , when they won an FA Cup and three of their First Division titles . The club was relegated in 2009 and again in 2016 . The club won promotion at the first time of asking each time , returning to the Premier League as Championship winners in 2010 and 2017 for the 2017-18 season . Newcastle has a local rivalry with Sunderland , with whom they have contested the Tyne-Wear derby since 1898 . The club 's traditional kit colours are black and white striped shirts , black shorts and black socks . Their crest has elements of the city coat of arms , which features two grey seahorses . Before each home game , the team enters the field to Local Hero , and Blaydon Races is also sung during games ."]
  ]
}
*/
question : When was the third highest paid Rangers F.C . player born ?.
The answer is : f_passage([Jean-Alain Boumsong])

/*
table segment = {
  "table_caption": "tourism in england",
  "table_column_priority": [
    ["national rank", "1"],
    ["site", "tower of london"],
    ["location", "london"],
    ["visitor count ( 2009 )", "2,389,548"]
  ]
}
*/
/*
linked passages = {
  "linked_passage_title": [
    "Tower of London",
    "London"
  ],
  "linked_passage_context": [
    ["Tower of London", "The Tower of London , officially Her Majesty 's Royal Palace and Fortress of the Tower of London , is a historic castle located on the north bank of the River Thames in central London . It lies within the London Borough of Tower Hamlets , which is separated from the eastern edge of the square mile of the City of London by the open space known as Tower Hill . It was founded towards the end of 1066 as part of the Norman Conquest of England . The White Tower , which gives the entire castle its name , was built by William the Conqueror in 1078 and was a resented symbol of oppression , inflicted upon London by the new ruling elite . The castle was used as a prison from 1100 ( Ranulf Flambard ) until 1952 ( Kray twins ) , although that was not its primary purpose . A grand palace early in its history , it served as a royal residence . As a whole , the Tower is a complex of several buildings set within two concentric rings of defensive walls and a moat . There were several phases of expansion , mainly under kings Richard I , Henry III , and Edward I in the 12th and 13th centuries . The general layout established by the late 13th century remains despite later activity on the site . The Tower of London has played a prominent role in English history . It was besieged several times , and controlling it has been important to controlling the country . The Tower has served variously as an armoury , a treasury , a menagerie , the home of the Royal Mint , a public record office , and the home of the Crown Jewels of England ."],
    ["London", "London is the capital and largest city of England and of the United Kingdom . Standing on the River Thames in the south-east of England , at the head of its 50-mile ( 80 km ) estuary leading to the North Sea , London has been a major settlement for two millennia . Londinium was founded by the Romans . The City of London , London 's ancient core \u2212 an area of just 1.12 square miles ( 2.9 km2 ) and colloquially known as the Square Mile \u2212 retains boundaries that closely follow its medieval limits . [ note 1 ] The City of Westminster is also an Inner London borough holding city status . Greater London is governed by the Mayor of London and the London Assembly . [ note 2 ] London is considered to be one of the world 's most important global cities and has been termed the world 's most powerful , most desirable , most influential , most visited , most expensive , innovative , sustainable , most investment friendly , and most popular for work city . London exerts a considerable impact upon the arts , commerce , education , entertainment , fashion , finance , healthcare , media , professional services , research and development , tourism and transportation . London ranks 26th out of 300 major cities for economic performance . It is one of the largest financial centres and has either the fifth or the sixth largest metropolitan area GDP . [ note 3 ] It is the most-visited city as measured by international arrivals and has the busiest city airport system as measured by passenger traffic . It is the leading investment destination , hosting more international retailers and ultra high-net-worth individuals than any other city ."]
  ]
}
*/
question : Which conquerors established the historical site in England that attracted 2,389,548 2009 tourists ?.
The answer is : f_passage([Tower of London])

/*
table segment = {
  "table_caption": "2006 league of ireland premier division",
  "table_column_priority": [
    ["team", "bray wanderers"],
    ["manager", "eddie gormley"],
    ["main sponsor", "slevin group"],
    ["kit supplier", "adidas"],
    ["stadium", "carlisle grounds"],
    ["capacity", "7,000"]
  ]
}
*/
/*
linked passages = {
  "linked_passage_title": [
    "Bray Wanderers A.F.C.",
    "Eddie Gormley",
    "Adidas",
    "Carlisle Grounds"
  ],
  "linked_passage_context": [
    ["Bray Wanderers A.F.C.", "Bray Wanderers Football Club ( Irish : Cumann Peile F\u00e1naithe Bhr\u00e9 ) are an Irish association football club playing in the League of Ireland First Division . The club in its present form was founded in 1942 in Bray , and was known until 2010 as Bray Wanderers A.F.C . It was elected to the League in 1985 , and plays its home matches at the Carlisle Grounds . Club colours are Green and White , and it goes by the nickname The Seagulls ."],
    ["Eddie Gormley", "Eddie Gormley ( born 23 October 1968 ) is an Irish football coach and former player who is currently manager of Cabinteely ."],
    ["Adidas", "Adidas AG ( German : [ \u02c8\u0294adi\u02ccdas ] AH-dee-DAHS ; stylized as \u0251did\u0251s since 1949 ) is a multinational corporation , founded and headquartered in Herzogenaurach , Germany , that designs and manufactures shoes , clothing and accessories . It is the largest sportswear manufacturer in Europe , and the second largest in the world , after Nike . It is the holding company for the Adidas Group , which consists of the Reebok sportswear company , 8.33% of the German football club Bayern Munich , and Runtastic , an Austrian fitness technology company . Adidas ' revenue for 2018 was listed at \u20ac21.915 billion . The company was started by Adolf Dassler in his mother 's house ; he was joined by his elder brother Rudolf in 1924 under the name Dassler Brothers Shoe Factory . Dassler assisted in the development of spiked running shoes ( spikes ) for multiple athletic events . To enhance the quality of spiked athletic footwear , he transitioned from a previous model of heavy metal spikes to utilising canvas and rubber . Dassler persuaded U.S. sprinter Jesse Owens to use his handmade spikes at the 1936 Summer Olympics . In 1949 , following a breakdown in the relationship between the brothers , Adolf created Adidas , and Rudolf established Puma , which became Adidas ' business rival . Adidas ' logo is three stripes , which is used on the company 's clothing and shoe designs as a marketing aid . The branding , which Adidas bought in 1952 from Finnish sports company Karhu Sports , became so successful that Dassler described Adidas as The three stripes company ."],
    ["Carlisle Grounds", "The Carlisle Grounds is a football stadium in Bray , County Wicklow , Ireland . Situated directly behind the Bray D.A.R.T . station , it is home to Bray Wanderers A.F.C . Its current capacity is roughly 4,000 ."]
  ]
}
*/
question : The home stadium of the Bray Wanderers of 2006 League of Ireland is situated behind what station ?.
The answer is : f_passage([Carlisle Grounds])

/*
table segment = {
  "table_caption": "list of stratigraphic units with ornithischian tracks",
  "table_column_priority": [
    ["name", "cerroa del pueblo formation"],
    ["location", "mexico"],
    ["description", "description"]
  ]
}
*/
/*
linked passages = {
  "linked_passage_title": [
    "Cerroa del Pueblo Formation",
    "Mexico"
  ],
  "linked_passage_context": [
    ["Cerroa del Pueblo Formation", "The Cerro del Pueblo Formation is a geological formation in Coahuila , Mexico whose strata date back to the Late Cretaceous . Dinosaur remains are among the fossils that have been recovered from the formation . The formation has been dated to between 72.5 Ma and 71.4 million years old ."],
    ["Mexico", "Mexico ( Spanish : M\u00e9xico [ \u02c8mexiko ] ( listen ) ; Nahuatl languages : M\u0113xihco ) , officially the United Mexican States ( UMS ; Spanish : Estados Unidos Mexicanos , EUM [ es\u02c8ta\u00f0os u\u02c8ni\u00f0oz mexi\u02c8kanos ] ( listen ) , lit . Mexican United States ) , is a country in the southern portion of North America . It is bordered to the north by the United States ; to the south and west by the Pacific Ocean ; to the southeast by Guatemala , Belize , and the Caribbean Sea ; and to the east by the Gulf of Mexico . Covering almost 2,000,000 square kilometers ( 770,000 sq mi ) , the nation is the fifth largest country in the Americas by total area and the 13th largest independent state in the world . With an estimated population of over 129 million people , Mexico is the tenth most populous country and the most populous Spanish-speaking country in the world , while being the second most populous nation in Latin America after Brazil . Mexico is a federation comprising 31 states plus Mexico City ( CDMX ) , which is the capital city and its most populous city . Other metropolises in the country include Guadalajara , Monterrey , Puebla , Toluca , Tijuana , and Le\u00f3n . Pre-Columbian Mexico dates to about 8000 BC and is identified as one of six cradles of civilization and was home to many advanced Mesoamerican civilizations such as the Olmec , Toltec , Teotihuacan , Zapotec , Maya , and Aztec before first contact with Europeans . In 1521 , the Spanish Empire conquered and colonized the territory from its politically powerful base in Mexico-Tenochtitlan ( part of Mexico City ) , which was administered as the viceroyalty of New Spain . The Roman Catholic Church played a powerful role in governing the country as millions were converted to the faith , although King Charles III expelled the Jesuits in the 1770s . The territory became a nation state following its recognition in 1821 after the Mexican War of Independence . The post-independence period was tumultuous , characterized by economic inequality and many contrasting political changes ."]
  ]
}
*/
question : Which Ornithischian formation is in a country bordered by Guatemala and Belize ?.
The answer is : f_passage([Mexico])

/*
table segment = {
  "table_caption": "nypd blue (season 2)",
  "table_column_priority": [
    ["actor", "nicholas turturro"],
    ["character", "james martinez"],
    ["main cast", "entire season"],
    ["recurring cast", "n/a"]
  ]
}
*/
/*
linked passages = {
  "linked_passage_title": [
    "James Martinez (NYPD Blue)",
    "Nicholas Turturro"
  ],
  "linked_passage_context": [
    ["James Martinez (NYPD Blue)", "James Martinez was a fictional character in the television series NYPD Blue . He was played by Nicholas Turturro from Seasons 1 to 7 ."],
    ["Nicholas Turturro", "Nicholas Turturro ( born January 29 , 1962 ) is an American actor , known for his roles in New York City based films and on the television series Blue Bloods and NYPD Blue . Nicholas is the younger brother of John Turturro and the cousin of Aida Turturro ."]
  ]
}
*/
question : What is the NYPD Blue character of the actor who was born on January 29 , 1962 ?.
The answer is : f_passage([Nicholas Turturro])"""

select_row_wise_three_shot_prompt = """Using f_row() api to select relevant rows in the given table and linked passages that support or oppose the question.
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
- alan hutton: alan hutton ( born 30 november 1984 ) is a scottish former professional footballer , who played as a right back . hutton started his career with rangers , and won the league title in 2005 . he moved to english football with tottenham hotspur in 2008 , and helped them win the league cup later that year . after a loan spell with sunderland in 2010 , hutton joined aston villa in 2011 . he initally struggled with villa , and was loaned to mallorca , nottingham forest and bolton wanderers . hutton then enjoyed a renaissance with villa , and left the club after helping them win promotion in 2019 . hutton made 50 appearances for the scotland national team between 2007 and 2016 .
- tottenham hotspur f.c.: tottenham hotspur football club , commonly referred to as tottenham ( /ˈtɒtənəm/ ) or spurs , is an english professional football club in tottenham , london , that competes in the premier league . tottenham hotspur stadium has been the club 's home ground since april 2019 , replacing their former home of white hart lane , which had been demolished to make way for the new stadium on the same site . their training ground is on hotspur way in bulls cross in the london borough of enfield . the club is owned by enic group . tottenham have played in a first ( home ) strip of white shirts and navy blue shorts since the 1898-99 season . the club 's emblem is a cockerel standing upon a football , with a latin motto audere est facere ( to dare is to do ) . founded in 1882 , tottenham won the fa cup for the first time in 1901 , the only non-league club to do so since the formation of the football league in 1888 . tottenham were the first club in the 20th century to achieve the league and fa cup double , winning both competitions in the 1960-61 season . after successfully defending the fa cup in 1962 , in 1963 they became the first british club to win a uefa club competition - the european cup winners ' cup . they were also the inaugural winners of the uefa cup in 1972 , becoming the first british club to win two different major european trophies . they have collected at least one major trophy in each of the six decades from the 1950s to 2000s - an achievement only matched by manchester united . in total , spurs have won two league titles , eight fa cups , four league cups , seven fa community shields , one european cup winners ' cup and two uefa cups .
passages linked to row 2
- giovanni van bronckhorst: giovanni christiaan van bronckhorst oon ( dutch pronunciation : [ ɟijoːˈvɑni vɑm ˈbrɔŋkɦɔrst ] ( listen ) ; born 5 february 1975 ) , also known by his nickname gio , is a retired dutch footballer and currently the manager of guangzhou r & f . formerly a midfielder , he moved to left back later in his career . during his club career , van bronckhorst played for rkc waalwijk , feyenoord , rangers , arsenal , barcelona and again with feyenoord . he was an instrumental player in barcelona 's 2005-06 uefa champions league victory , being in the starting line-up of the final , having played every champions league match for barcelona that season . van bronckhorst earned 107 caps for the netherlands national team , and played for his country in three fifa world cups , in 1998 , 2006 and 2010 , as well as three uefa european championships , in 2000 , 2004 and 2008 . after captaining the oranje in the 2010 world cup final , he was elected into the order of orange-nassau . the 2010 world cup final was the last match in his career . after assisting the dutch under-21 team and feyenoord , van bronckhorst became feyenoord manager in may 2015 . he won the knvb cup in his first season and the club 's first eredivisie title for 18 years in 2017 .
- arsenal f.c.: arsenal football club is a professional football club based in islington , london , england , that plays in the premier league , the top flight of english football . the club has won 13 league titles , a record 13 fa cups , 2 league cups , 15 fa community shields , 1 league centenary trophy , 1 uefa cup winners ' cup and 1 inter-cities fairs cup . arsenal was the first club from the south of england to join the football league , in 1893 , and they reached the first division in 1904 . relegated only once , in 1913 , they continue the longest streak in the top division , and have won the second-most top-flight matches in english football history . in the 1930s , arsenal won five league championships and two fa cups , and another fa cup and two championships after the war . in 1970-71 , they won their first league and fa cup double . between 1989 and 2005 , they won five league titles and five fa cups , including two more doubles . they completed the 20th century with the highest average league position . herbert chapman won arsenal 's first national trophies , but died prematurely . he helped introduce the wm formation , floodlights , and shirt numbers , and added the white sleeves and brighter red to the club 's kit . arsène wenger was the longest-serving manager and won the most trophies . he won a record 7 fa cups , and his title-winning team set an english record for the longest top-flight unbeaten league run at 49 games between 2003 and 2004 , receiving the nickname the invincibles .
passages linked to row 3
- jean-alain boumsong: jean-alain boumsong somkong ( born 14 december 1979 ) is a former professional football defender , including french international . he is known for his physical strength , pace and reading of the game .
- newcastle united f.c.: newcastle united football club is an english professional football club based in newcastle upon tyne , tyne and wear , that plays in the premier league , the top tier of english football . founded in 1892 by the merger of newcastle east end and newcastle west end . per the requirements from the taylor report , in response to the hillsborough disaster , that all premiership teams have an all-seater stadium , the grounds were adjusted in the mid-1990s and now has a capacity of 52,354. , the team plays its home matches at st james ' park . the club has been a member of the premier league for all but three years of the competition 's history , spending 86 seasons in the top tier as of may 2018 , and has never dropped below english football 's second tier since joining the football league in 1893 . they have won four league championship titles , six fa cups and a charity shield , as well as the 1969 inter-cities fairs cup and the 2006 uefa intertoto cup , the ninth highest total of trophies won by an english club . the club 's most successful period was between 1904 and 1910 , when they won an fa cup and three of their first division titles . the club was relegated in 2009 and again in 2016 . the club won promotion at the first time of asking each time , returning to the premier league as championship winners in 2010 and 2017 for the 2017-18 season . newcastle has a local rivalry with sunderland , with whom they have contested the tyne-wear derby since 1898 . the club 's traditional kit colours are black and white striped shirts , black shorts and black socks . their crest has elements of the city coat of arms , which features two grey seahorses . before each home game , the team enters the field to local hero , and blaydon races is also sung during games .
passages linked to row 4
- carlos cuéllar: carlos javier cuéllar jiménez ( spanish pronunciation : [ ˈkaɾlos ˈkweʎaɾ ] ; born 23 august 1981 ) is a spanish professional footballer who plays for israeli club bnei yehuda . mainly a central defender , he can also operate as a right back .
- aston villa: aston villa football club ( nicknamed villa ) is an english professional football club based in aston , birmingham . the club competes in the premier league , the top tier of the english football league system . founded in 1874 , they have played at their home ground , villa park , since 1897 . aston villa were one of the founder members of the football league in 1888 and of the premier league in 1992 . villa are one of only five english clubs to have won the european cup , in 1981-82 . they have also won the football league first division seven times , the fa cup seven times , the league cup five times , and the uefa super cup once . villa have a fierce local rivalry with birmingham city and the second city derby between the teams has been played since 1879 . the club 's traditional kit colours are claret shirts with sky blue sleeves , white shorts and sky blue socks . their traditional club badge is of a rampant lion . the club is currently owned by the nswe group , a company owned by the egyptian billionaire nassef sawiris and the american billionaire wes edens .
*/

question : when was the third highest paid rangers f.c . player born ?
The answer is : f_row([row 3])

/*
table caption : missouri valley conference men's basketball tournament
col : year | mvc champion | score | runner-up | tournament mvp | venue ( and city )
row 1 : 1994 | southern illinois | 77-74 | northern iowa | cam johnson , northern iowa | st. louis arena ( st. louis , missouri )
row 2 : 1996 | tulsa | 60-46 | bradley | shea seals , tulsa | kiel center ( st. louis , missouri )
*/

/*
passages linked to row 1
- southern illinois salukis men's basketball: the southern illinois salukis men 's basketball team represents southern illinois university carbondale in carbondale , illinois . the salukis compete in the missouri valley conference , and they play their home games at banterra center . as of march 2019 , saluki hall of fame basketball player , bryan mullins , has become the newest head coach of the southern illinois basketball program .
- northern iowa panthers men's basketball: the northern iowa panthers men 's basketball team represents the university of northern iowa , located in cedar falls , iowa , in ncaa division i basketball competition . uni is currently a member of the missouri valley conference .
passages linked to row 2
- tulsa golden hurricane men's basketball: the tulsa golden hurricane men 's basketball team represents the university of tulsa in tulsa , in the u.s. state of oklahoma . the team participates in the american athletic conference . the golden hurricane hired frank haith from missouri on april 17 , 2014 to replace danny manning , who had resigned to take the wake forest job after the 2013-14 season . the team has long been successful , especially since the hiring of nolan richardson in 1980 . many big-name coaches previously worked at tulsa , like university of kansas coach bill self and minnesota coach tubby smith . the hurricane have been to the ncaa tournament 14 times in their history . in addition , they have won two national invitation tournaments , in 1981 and 2001 , and one cbi tournament . in 2005 , street & smith 's named the university of tulsa as the 59th best college basketball program of all time .
- bradley braves men's basketball: the bradley braves men 's basketball team represents bradley university , located in peoria , illinois , in ncaa division i basketball competition . they compete as a member of the missouri valley conference . the braves are currently coached by brian wardle and play their home games at carver arena . bradley has appeared in nine ncaa tournaments , including two final fours , finishing as the national runner-up in 1950 and 1954 . they last appeared in the ncaa tournament in 2019 , and last reached the ncaa sweet sixteen in 2006 . the braves have also appeared in the national invitation tournament 21 times with an all-time nit record of 26-18 and have won four nit championships ( 1957 , 1960 , 1964 , and 1982 ) , second only to st. john 's in appearances ( 30 ) and titles ( 5 ) . until the introduction of the vegas 16 tournament in 2016 , the program held the distinction of being invited to the initial offering of every national postseason tournament .
*/

question : how tall , in feet , is the basketball personality that was chosen as mvp most recently ?
The answer is : f_row([row 2])

/*
table caption : list of longest - serving soap opera actors.
col : dance | best dancer | best score | worst dancer | worst score
row 1 : boogie woogie | kaspar capparoni | 44 | barbara capponi | 27
row 2 : merengue | gedeon burkhard | 36 | paolo rossi | 25
row 3 : quickstep | kaspar capparoni | 44 | alessandro di pietro | 9
row 4 : samba | gedeon burkhard | 39 | giuseppe povia | 20
row 5 : tango | sara santostasi | 40 | gedeon burkhard | 27
*/

/*
passages linked to row 1
Title: kaspar capparoni. Content: gaspare kaspar capparoni ( born 1 august 1964 ) is an italian actor .
Title: barbara capponi. Content: the seventh series of ballando con le stelle was broadcast from 26 february to 30 april 2011 on rai 1 and was presented by milly carlucci with paolo belli and his 'big band ' .
passages linked to row 2
Title: gedeon burkhard. Content: gedeon burkhard ( born 3 july 1969 ) is a german film and television actor . although he has appeared in numerous films and tv series in both europe and the us , he is probably best recognised for his role as alexander brandtner in the austrian/german television series kommissar rex ( 1997-2001 ) , which has been aired on television in numerous countries around the world , or as corporal wilhelm wicki in the 2009 film inglourious basterds . he is also well recognised for his role as chris ritter in the long-running series alarm für cobra 11 .
Title: paolo rossi. Content: paolo rossi ( italian pronunciation : [ ˈpaːolo ˈrossi ] ; born 23 september 1956 ) is an italian former professional footballer , who played as a forward . in 1982 , he led italy to the 1982 fifa world cup title , scoring six goals to win the golden boot as top goalscorer , and the golden ball for the player of the tournament . rossi is one of only three players to have won all three awards at a world cup , along with garrincha in 1962 , and mario kempes in 1978 . rossi was also awarded the 1982 ballon d'or as the european footballer of the year for his performances . along with roberto baggio and christian vieri , he is italy 's top scorer in world cup history , with nine goals in total . at club level , rossi was also a prolific goalscorer for vicenza . in 1976 , he was signed to juventus from vicenza in a co-ownership deal for a world record transfer fee . vicenza retained his services , and he was top goalscorer in serie b in 1977 , leading his team to promotion to serie a . the following season , rossi scored 24 goals , to become the first player to top the scoring charts in serie b and serie a in consecutive seasons . in 1981 rossi made his debut for juventus , and went on to win two serie a titles , the coppa italia , the uefa cup winners ' cup , the uefa super cup , and the european cup . widely regarded as one of the greatest italian footballers of all time , rossi was named in 2004 by pelé as one of the top 125 greatest living footballers as part of fifa 's 100th anniversary celebration . in the same year , rossi placed 12 in the uefa golden jubilee poll .
passages linked to row 3
Title: kaspar capparoni. Content: gaspare kaspar capparoni ( born 1 august 1964 ) is an italian actor .
Title: alessandro di pietr. Content: the seventh series of ballando con le stelle was broadcast from 26 february to 30 april 2011 on rai 1 and was presented by milly carlucci with paolo belli and his 'big band ' .
passages linked to row 4
Title: gedeon burkhard. Content: gedeon burkhard ( born 3 july 1969 ) is a german film and television actor . although he has appeared in numerous films and tv series in both europe and the us , he is probably best recognised for his role as alexander brandtner in the austrian/german television series kommissar rex ( 1997-2001 ) , which has been aired on television in numerous countries around the world , or as corporal wilhelm wicki in the 2009 film inglourious basterds . he is also well recognised for his role as chris ritter in the long-running series alarm für cobra 11 .
Title: giuseppe povia. Content: giuseppe povia ( italian pronunciation : [ poˈviːa ] } , born november 19 , 1972 ) , better known just as povia [ ˈpɔːvja ] , is an italian rock singer-songwriter .
passages linked to row 5
Title: sara santostasi. Content: sara santostasi ( born 24 january 1993 ) is an italian actress singer and dancer . she was one of the contestants in seventh series of ballando con le stelle , the italian version of dancing with the stars .
Title: gedeon burkhard. Content: gedeon burkhard ( born 3 july 1969 ) is a german film and television actor . although he has appeared in numerous films and tv series in both europe and the us , he is probably best recognised for his role as alexander brandtner in the austrian/german television series kommissar rex ( 1997-2001 ) , which has been aired on television in numerous countries around the world , or as corporal wilhelm wicki in the 2009 film inglourious basterds . he is also well recognised for his role as chris ritter in the long-running series alarm für cobra 11 .
*/

question : what is the highest best score series 7 of ballando con le stelle for the best dancer born 3 july 1969 ?
The answer is : f_row([row 4])"""