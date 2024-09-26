# 🌙 LunaCARE - Empowering Women’s Menstrual Health & Well-being

Welcome to **LunaCare**, a comprehensive platform designed to support women in managing their menstrual health, offering personalized tools, educational content, and a supportive community. Whether 
navigating regular cycles, dealing with Polycystic Ovary Syndrome (PCOS), or 
transitioning into menopause, LunaCare offers a personalized, supportive experience for 
women at every stage of life.

---
<div align="center">
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/192158954-f88b5814-d510-4564-b285-dff7d6400dad.png" alt="HTML" title="HTML"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/183898674-75a4a1b1-f960-4ea9-abcb-637170a00a75.png" alt="CSS" title="CSS"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/117447155-6a868a00-af3d-11eb-9cfe-245df15c9f3f.png" alt="JavaScript" title="JavaScript"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" alt="Python" title="Python"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/183423775-2276e25d-d43d-4e58-890b-edbc88e915f7.png" alt="Flask" title="Flask"/></code>
	<code><img width="50" src="https://github-production-user-asset-6210df.s3.amazonaws.com/136815194/253220886-02494c7c-de6a-43a6-9293-6369696842ed.png" alt="Canva" title="Canva"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/183570228-6a040b9f-3ddf-47a2-a201-743121dac664.png" alt="php" title="php"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/183896128-ec99105a-ec1a-4d85-b08b-1aa1620b2046.png" alt="MySQL" title="MySQL"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/192108891-d86b6220-e232-423a-bf5f-90903e6887c3.png" alt="Visual Studio Code" title="Visual Studio Code"/></code>
	<code><img width="50" src="https://user-images.githubusercontent.com/25181517/183914128-3fc88b4a-4ac1-40e6-9443-9a30182379b7.png" alt="Jupyter Notebook" title="Jupyter Notebook"/></code>
	<code><img width="50" src="https://github.com/marwin1991/profile-technology-icons/assets/76012086/4ec200c2-acdf-4c42-b419-cd49cba3d09f" alt="NumPy" title="NumPy"/></code>
	<code><img width="50" src="https://github.com/marwin1991/profile-technology-icons/assets/76012086/24b02d77-2f28-43c7-b5d6-e15e3395851b" alt="Pandas" title="Pandas"/></code>
</div>

---


https://github.com/user-attachments/assets/49dc4633-c670-4f90-a383-e55e3e59d933


## 🏷️ What Does LunaCare Stand For?

- **Luna**: Derived from the Latin word for the moon, symbolizing the natural cycles that mirror a woman's body, just like lunar phases.
- **Care**: Represents the compassion, support, and personalized resources LunaCare provides to ensure every woman feels empowered in her health journey.



## 💡 Why LunaCare?

Women's menstrual health is a critical aspect of overall well-being, yet it remains an 
often overlooked area in healthcare. Many women struggle to find reliable, accurate, and 
accessible information regarding their menstrual cycles, leading to confusion and potential 
health risks. Conditions such as Polycystic Ovary Syndrome (PCOS) are prevalent, but 
they are frequently misunderstood, underdiagnosed, or mismanaged. This gap in 
understanding can result in women feeling overwhelmed and unsupported in managing 
their symptoms, which further complicates their healthcare journey.

## 🌚 Why Choose the Moon as Symbolic Referance?

The moon, with its cyclical phases, has long been a symbol of feminine energy and natural rhythms. Just as the moon waxes and wanes, so too does the menstrual cycle. This celestial body, with its mysterious glow and profound influence on tides, serves as a powerful metaphor for the ebb and flow of a woman's body and emotions. By aligning our menstrual care website with the themes of the moon, we aim to create a space that is both informative and empowering, celebrating the natural beauty and wisdom of the female experience.

## 🌟 What LunaCare Offers

LunaCare is an all-in-one platform for menstrual health management, combining education, tracking tools, and PCOS detection capabilities.

### Key Features:
- **📚 Educational Content**: Evidence-based articles and expert insights on various menstrual health topics, including cycle management, PCOS, and lifestyle tips.
- **📅 Symptom and Cycle Tracking Tools**: Track your menstrual cycle, daily symptoms, mood, and overall well-being. Get accurate predictions for your next period or ovulation.
- **🩺 PCOS Screening and Tracking**: Identify and manage PCOS symptoms with our algorithmic assessment and symptom tracking.
- **🤖 Chatbot Assistance**: Get real-time help and advice through our natural language processing chatbot for menstrual health queries.
- **🛍️ Product Comparison Using Sentiment Analysis**: Compare menstrual products based on customer reviews and sentiment analysis to make informed purchasing decisions.
- **✍️ Blog Pages**: Read expert advice, personal stories, and educational content related to menstrual health, and engage with the LunaCare community.


## 1️⃣ PCOS Detector

The **PCOS Detector** utilizes data collected from user inputs and validated reports from **Mayo Clinic** to assess the risk of Polycystic Ovary Syndrome (PCOS). The tool is built upon a machine learning model trained on various algorithms, with **XGBoost** providing the most accurate results.

- **User Inputs**: Height, weight, average cycle length, acne occurrence, hair loss, etc.
- **Algorithms Used**: Logistic Regression, Decision Trees, Random Forest, Gradient Boost, XGBoost, CatBoost etc.
- **Best Performing Model**: Support Vector (used in the final web project)
- **Risk Evaluation**: Considers hormonal imbalances, symptom history, and clinical findings
- **Medical Verification**: All data and results are verified by medical professionals
- **Data Storage**: Results are stored in a database for future model training and improvement



https://github.com/user-attachments/assets/33aea468-4b5c-4045-950a-593bc1bcaada


## 2️⃣ Period Tracker

The **Period Tracker** allows users to input their menstrual cycle information and calculates the expected dates for future periods. This tool also predicts ovulation days and provides a view of upcoming menstrual phases for the next five months.

- **User Inputs**: Average cycle length, period cycle length, last period date
- **Formula**:
  - Next Period Date:
    ```bash
    Next Period Date = Last Period Date + Average Cycle Length
    ```
  - Ovulation Day:
    ```bash
    Ovulation Day = Last Period Date + (Average Cycle Length - 14)
    ```
- **Output**: Displays a calendar showing probable period dates, ovulation days, pre-period, and post-period days for the next five months.



https://github.com/user-attachments/assets/87f71102-7436-4c74-9d1d-e238322f8650



## 3️⃣ Blogs

LunaCare’s **Blog Section** features well-researched articles focused on various menstrual health topics, including PCOS management, menstrual cycle education, and lifestyle advice. The blog content is enriched with interactive visual data to make information more accessible and engaging.

- **Topics Covered**: Menstrual health, PCOS management, lifestyle tips etc.
- **Data-Driven Insights**: Articles include data analysis and Python-generated visualizations
- **Interactive Dashboards**: Integrated with Tableau for more detailed insights and user engagement



https://github.com/user-attachments/assets/fbaa99dc-7b99-42bc-a7e2-644169b6fbd1



## 4️⃣ Product Comparison

The **Product Comparison** tool uses real-time web scraping using `BeautifulSoup` and `Selenium` from **Amazon** to analyze reviews and ratings of menstrual health products. Sentiment analysis is performed to help users compare and choose the best products.

- **Web Scraping**: Uses ASIN numbers to scrape product reviews and details from Amazon
- **Sentiment Analysis**: Performed on product reviews to compare user satisfaction
- **Comparison Output**:
  - **Price Comparison**
  - **Overall Sentiment**: Positive/Negative/Neutral
  - **Product Features**: Highlights pros and cons
- **Direct Purchase Link**: Users can directly purchase the compared products via Amazon links



https://github.com/user-attachments/assets/0eee038e-94d5-46bc-a1f5-e009ebea76e0



## 5️⃣ Chatbot

LunaCare features a **Chatbot** designed to answer frequently asked questions related to menstrual health. The chatbot is trained on over **500+ questions** and interacts with users to provide personalized information and guidance.

- **Question Bank**: 500+ questions related to periods, PCOS, and general menstrual health
- **Training Data**: JSON-based static dataset used for training
- **Capabilities**:
  - Symptom logging
  - Period tracking
  - Access to educational content
- **Real-Time Interaction**: Offers immediate answers to user questions, improving user experience


https://github.com/user-attachments/assets/a822167e-966c-40c7-b84b-727b20584c61


## 6️⃣ Testimonials

Customer testimonials get saved and dynamically displayed on the webpage. Filling the form available on the Home Page stores the testimonials in the database. This is retrieved and displayed on the page. If the administrator decides to delete a testimonial, it gets reflected on the main page from where it is removed as well.

## 🌐 System Design

### Front-End Technologies

#### HTML (HyperText Markup Language) 🏷️
- **Content Structure**: Provides a well-defined structure for the content, ensuring that users can easily navigate and access information.
- **Accessibility**: Supports accessibility features, making the website more inclusive for users with disabilities.
- **SEO Friendly**: Proper use of HTML tags helps improve search engine optimization (SEO), making the website more discoverable.

#### CSS (Cascading Style Sheets) 🎨
- **Consistent Design**: Ensures a consistent and visually appealing design across all pages.
- **Responsive Design**: Enables the implementation of responsive design principles, ensuring the website looks and functions well on various devices (desktops, tablets, smartphones).
- **Customization**: Allows extensive customization, enabling LunaCare to establish a unique brand identity through the website's look and feel.

#### JavaScript 🌐
- **Interactivity**: Adds interactivity to the website, such as interactive tools (e.g., period cycle calculator, PCOS prediction tool) and real-time updates.
- **Dynamic Content**: Enables fetching and displaying data dynamically, enhancing the user experience by providing up-to-date information without requiring page reloads.
- **User Engagement**: Allows for the implementation of interactive elements like forms, quizzes, and chatbots, making the website more engaging for users.


### Back-End Technologies

#### Flask (Python Framework) 🐍
- **Lightweight**: Flask is a micro-framework that is lightweight and easy to set up, making it suitable for small to medium-sized applications like LunaCare.
- **Flexibility**: Offers flexibility to structure the application as needed, with minimal overhead.
- **Extensible**: Easily extensible with various libraries and tools, allowing integration of additional features as required.

#### SQLAlchemy (ORM for Python) 🛢️
- **ORM Features**: Provides Object-Relational Mapping (ORM) features, allowing developers to interact with databases using Python objects instead of writing raw SQL queries.
- **Database-Agnostic**: Works with multiple database systems, providing flexibility in database management.
- **Efficiency**: Simplifies complex database operations, reducing the amount of boilerplate code and improving development efficiency.

#### MySQL 🏠
- **Relational Database**: Provides structured storage and efficient querying capabilities for relational data, making it suitable for managing user data, forum posts, and other structured content.
- **Scalability**: Can handle large volumes of data and concurrent transactions, ensuring scalability as the user base grows.
- **Reliability**: Offers robust transaction support and data integrity features, ensuring reliable data storage. Also supports XAMPP connections.

### 🎨 Graphic Design Tools

#### Canva
- **User-Friendly**: Easy-to-use interface that allows for quick design creation.
- **Templates**: Provides a variety of templates that can be customized to match LunaCare’s brand identity.
- **Collaboration**: Enables team collaboration, ensuring consistency and quality across all graphic elements.

## 🗃️ Documentation

- **Find All Details Here**: [REPORT](https://github.com/AsmitaMondal/LunaCARE/blob/main/Report/2348018_Web%20Project.pdf)


## ❤️ Made With Love By

- Asmita Roy Mondal
- Sayan Pal
- Swarnasish Banerjee
