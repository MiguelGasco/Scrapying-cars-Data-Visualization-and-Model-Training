# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import scrapy
import re
from Cars.items import CarsItem

class CarsPipeline:
    def process_item(self, item, spider):
        if isinstance(item, CarsItem):
            item['price'] = re.sub(r'[^0-9]', '', item['price'].strip())
            item['Mileage'] = re.sub(r'[^0-9]', '', item['Mileage'].strip())
            item['Gearbox'] = item['Gearbox'].strip()
            item['Initial_registration'] = re.sub(r'[a-zA-Z]', '', item['Initial_registration'].strip())
            item['Fuel'] = item['Fuel'].strip()
            item['Power'] =item['Power'].strip()
            item['Seller'] = item['Seller'].strip()

            return item
        return item