url_pattern = 'https://www.hani.co.kr/arti/science/future/gallery{}.html'
img_pattern1 = '//*[@id="section-left-scroll-in"]/div[3]/div[{}]/li[{}]/div/span/a/img'
img_pattern2 = '//*[@id="section-left-scroll-in"]/div[3]/div[8]/li/div/span/a/img'
for i in range(1,64) :
    url = url_pattern.format(i); #url은 1부터 63까지
    # 1번 이미지 : //*[@id="section-left-scroll-in"]/div[3]/div[1]/li[1]/div/span/a/img
    # 2번 이미지 : //*[@id="section-left-scroll-in"]/div[3]/div[1]/li[2]/div/span/a/img
    #3번 이미지 : //*[@id="section-left-scroll-in"]/div[3]/div[2]/li[1]/div/span/a/img
    #4번 이미지 : //*[@id="section-left-scroll-in"]/div[3]/div[2]/li[2]/div/span/a/img
    #5번 이미지 : //*[@id="section-left-scroll-in"]/div[3]/div[3]/li[1]/div/span/a/img
    #14번 이미지 : //*[@id="section-left-scroll-in"]/div[3]/div[7]/li[2]/div/span/a/img
    for j in range (1,3) :
        for k in range (1,8) :
            try:
                driver.get(url);
                img = img_pattern1.format(k,j)
                imgLink = driver.find_element(By.XPATH, img);
                imgLink.click()
                driver.implicitly_wait(0.5)
                current_url = driver.current_url
                driver.get(current_url)
                title = driver.find_element(By.XPATH, "//*[@id='article_view_headline']/h4/span")
                date = driver.find_element(By.XPATH, "//*[@id='article_view_headline']/p[2]/span[1]")
                content = driver.find_element(By.XPATH, "//*[@id='a-left-scroll-in']/div[2]/div/div[2]")
                print(title.text)
                print(date.text)
                print(content.text)
            except Exception as e:
                print(e);
                continue
    driver.get(url)
    img = img_pattern2;
    imgLink = driver.find_element(By.XPATH, img);
    imgLink.click()
    driver.implicitly_wait(0.5)
    current_url = driver.current_url
    driver.get(current_url)
    title = driver.find_element(By.XPATH, "//*[@id='article_view_headline']/h4/span")
    date = driver.find_element(By.XPATH, "//*[@id='article_view_headline']/p[2]/span[1]")
    content = driver.find_element(By.XPATH, "//*[@id='a-left-scroll-in']/div[2]/div/div[2]")
    print(title.text)
    print(date.text)
    print(content.text)
