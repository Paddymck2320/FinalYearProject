import NFL_Scraper as spr

year = 2021
week = 18
multiple_weeks = True

spr.readData(year=year,week=week,multiple_weeks=multiple_weeks)
spr.buildAggregate(year=year,week=week,multiple_weeks=multiple_weeks)