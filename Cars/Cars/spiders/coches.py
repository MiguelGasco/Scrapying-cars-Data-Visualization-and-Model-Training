import scrapy
from Cars.items import CarsItem

class CochesSpider(scrapy.Spider):
    
    name = "coches"
    allowed_domains = ["www.autoscout24.de"]
    start_urls = [
        f"https://www.autoscout24.de/lst/bmw/118?atype=C&cy=D&desc=0&ocs_listing=include&page={i}&search_id=duwtoss0f3&sort=standard&source=listpage_pagination&ustate=N%2CU"
        for i in range(21)
    ]

    def parse(self, response):
        for cars in response.css('article'):
            yield response.follow(url=cars.css('a::attr(href)').get(), callback=self.parse_detail)
        
        # Obtener la página siguiente
        next_page = response.css('li.pagination-item::attr(href)').get()
        if next_page:
            yield response.follow(url=next_page, callback=self.parse)

    def parse_detail(self, response):
        
        #Precio del coche
        for car in response.css('div.PriceInfo_wrapper__hreB_'):
            price = car.css('span.PriceInfo_price__XU0aF::text').get()
            
        #Div con los datos genericos
        for car in response.css('div.VehicleOverview_containerMoreThanFourItems__691k2'):
            Mileage = car.css('div.VehicleOverview_itemContainer__XSLWi:nth-child(1) > div:nth-child(4)::text').get()
            Gearbox = car.css('div.VehicleOverview_itemContainer__XSLWi:nth-child(2) > div:nth-child(4)::text').get()
            Initial_registration = car.css('div.VehicleOverview_itemContainer__XSLWi:nth-child(3) > div:nth-child(4)::text').get()
            Fuel = car.css('div.VehicleOverview_itemContainer__XSLWi:nth-child(4) > div:nth-child(4)::text').get()
            Power = car.css('div.VehicleOverview_itemContainer__XSLWi:nth-child(5) > div:nth-child(4)::text').get()
            Seller = car.css('div.VehicleOverview_itemContainer__XSLWi:nth-child(6) > div:nth-child(4)::text').get()


        #Propietarios
        owners_text = response.css('#listing-history-section > div:nth-child(1) > div:nth-child(2) > dl:nth-child(1) > dd:nth-child(8)::text').get()
        try:
            owners = int(owners_text) if owners_text else None
        except ValueError:
            owners = None
        
        #Color exterior
        color_exterior = response.css('#color-section > div:nth-child(1) > div:nth-child(2) > dl:nth-child(1) > dd:nth-child(2)::text').get()
        
        #Color interior
        color_interior = response.css('#color-section > div:nth-child(1) > div:nth-child(2) > dl:nth-child(1) > dd:nth-child(8)::text').get()
        
        #Div con los datos de confort
        for data in response.css('.DataGrid_asColumnUntilLg__HEguB > dd:nth-child(2)'):
            confort = data.css('ul > li ::text').getall()
        
        #Div con los datos de entertainment
        for data in response.css('.DataGrid_asColumnUntilLg__HEguB > dd:nth-child(4)'):
            entertainment = data.css('ul > li ::text').getall()

        #Div con los datos de security
        for data in response.css('.DataGrid_asColumnUntilLg__HEguB > dd:nth-child(6)'):
            security = data.css('ul > li ::text').getall()

        #Div con los datos de extras
        for data in response.css('.DataGrid_asColumnUntilLg__HEguB > dd:nth-child(8)'):
            extras = data.css('ul > li ::text').getall()


        item = CarsItem(
            price=price,
            Mileage=Mileage,
            Gearbox=Gearbox,
            Initial_registration=Initial_registration,
            Fuel=Fuel,
            Power=Power,
            Seller=Seller,
            Owners=owners,
            Color_interior=color_interior,
            Color_exterior=color_exterior,
            Comfort=confort,
            Entertainment=entertainment,
            Security=security,
            Extras=extras
        )

        yield item