from selenium import webdriver
import pandas as pd
import os


url = "http://locuszoom.org/genform.php?type=yourdata"
driver_path = ''
download_path = ""
if download_path[-1] != '\\':
    download_path += '\\'
class Auto_downloader:
    def __init__(self, url=url, driver_path=driver_path, download_path=download_path):
        options = webdriver.FirefoxOptions()
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.download.manager.showWhenStarting",False)
        options.set_preference("browser.download.dir", download_path)
        options.set_preference("browser.download.useDownloadDir", True)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/download")
        options.set_preference("pdfjs.disabled", True)
        # options.add_argument("--headless")
        kwargs = {'executable_path': driver_path,
                  'options': options}
        browser = webdriver.Firefox(**kwargs)
        browser.get(url)
        browser.find_element_by_id("pvalcol").send_keys("P_BOLT_LMM_INF")
        browser.find_element_by_id("markercol").send_keys("SNP")
        self.submit = browser.find_element_by_xpath("/html/body/form/table/tbody/tr[5]/td[1]/input[1]")
        self.browser = browser

    def __del__(self):
        self.browser.close()

    def generate_plot(self, file, snp_name):  # file: summary statistic file, snp_name: lead snp
        self.browser.find_element_by_id("datafile").clear()
        self.browser.find_element_by_id("datafile").send_keys(file)
        self.browser.find_element_by_id("snpname").clear()
        self.browser.find_element_by_id("snpname").send_keys(snp_name)
        self.submit.click()
