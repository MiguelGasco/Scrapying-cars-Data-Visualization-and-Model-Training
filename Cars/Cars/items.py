# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CarsItem(scrapy.Item):
    price : int = scrapy.Field()
    Mileage : int  = scrapy.Field()
    Gearbox : str = scrapy.Field()
    Initial_registration : str = scrapy.Field()
    Fuel : str = scrapy.Field()
    Power : str = scrapy.Field()
    Seller : str = scrapy.Field()
    Owners : int= scrapy.Field()
    Color_interior : str = scrapy.Field()
    Color_exterior : str = scrapy.Field()
    Comfort = scrapy.Field(default=list)
    Entertainment = scrapy.Field(default=list)
    Security = scrapy.Field(default=list)
    Extras = scrapy.Field(default=list)