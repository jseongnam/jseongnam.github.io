

jpg_xpath_pattern = '//*[@id="ulContentStr"]/li[{}]/a/div[2]/img' # 1~21
more_xpath_pattern = '//*[@id="aMore"]' # 더보기 버튼
wait = WebDriverWait(driver, 10);
# main page 경로는 다 똑같음
# jpg1 xpath : //*[@id="ulContentStr"]/li[1]/a/div[2]/img
# jpg2 xpath : //*[@id="ulContentStr"]/li[2]/a/div[2]/img
# jpg3 xpath : //*[@id="ulContentStr"]/li[3]/a/div[2]/img
# jpg4 xpath : //*[@id="ulContentStr"]/li[4]/a/div[2]/img
# jpg21 xpath : //*[@id="ulContentStr"]/li[21]/a/div[2]/img
# jpg22 xpath : //*[@id="ulContentStr"]/li[22]/a/div[2]/img
# jpg42 xpath : //*[@id="ulContentStr"]/li[42]/a/div[2]/img
# 더보기 xpath : //*[@id="aMore"]
# 1~21 보고
# 더보기 누르고
# 22~42 보고
# 더보기 누르고
# 43~63 보기
# error 발생할떄까지 무한반복
# title : //*[@id="PostView"]/div[1]/div[1]/div[2]/h3
# date : //*[@id="PostView"]/div[1]/div[1]/div[3]/div[1]/div[2]
# content : //*[@id="ContentView"]
driver.get('https://www.brainmedia.co.kr/MediaContent/MediaContentList.aspx?MenuCd=BrainScience');

reply = 0;
while 1 : 
    for i in range(1+reply*21,22+reply*21) : 
        try :
            driver.get('https://www.brainmedia.co.kr/MediaContent/MediaContentList.aspx?MenuCd=BrainScience');
            for _ in range(reply) : 
                more_Link = wait.until(EC.element_to_be_clickable((By.XPATH, more_xpath_pattern)))
                more_Link.click();
            
            img = jpg_xpath_pattern.format(i);
            img_Link = wait.until(EC.element_to_be_clickable((By.XPATH, img)))
            img_Link.click();
              
            title = wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='PostView']/div[1]/div[1]/div[2]/h3")))
            date = wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='PostView']/div[1]/div[1]/div[3]/div[1]/div[2]")))
            content = wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='ContentView']")))
            
            print(title.text);
            print(date.text);
            print(content.text);
            

        except Exception as e :
            continue
    reply += 1;
driver.quit()
