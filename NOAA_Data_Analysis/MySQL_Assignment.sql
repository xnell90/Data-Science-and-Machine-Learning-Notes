/*Question 1*/
SELECT
	name AS 'Location',
    COUNT(staid) AS '# Stations'
FROM
	location
LEFT JOIN stationbylocation ON stationbylocation.locid = location.locationid
GROUP BY
	location.locationid
HAVING
	COUNT(staid) >= 100
ORDER BY
	COUNT(staid);
/* This query lists how many stations are found in each location. It
only shows all locations that have 100 or more stations. Based on the
output, most locations that have over 100 stations are in the United
States.*/

/*Question 2*/
SELECT
	location.name 'Location Name',
    ROUND(MIN(station.elevation), 2) 'Minimum Elevation',
    ROUND(MAX(station.elevation), 2) 'Maximum Elevation',
    ROUND(AVG(station.elevation), 1) 'Average Elevation'
FROM
	location
LEFT JOIN stationbylocation ON stationbylocation.locid = location.locationid
LEFT JOIN station ON station.stationid = stationbylocation.staid
WHERE
	location.name IN (
		SELECT
			name AS 'Location'
		FROM
			location
		LEFT JOIN stationbylocation ON stationbylocation.locid = location.locationid
		GROUP BY
			locationid
		HAVING
			COUNT(staid) >= 100
    )
GROUP BY
	location.locationid
ORDER BY 4 DESC;
/* This query gives a list of location name, the minumum elevation of its stations,
the maximum elevation of its stations, and the average elevation of its stations.
It includes all locations with 100 or more stations and it ordered by average elevation
(descending order). Based on the results, the top 3 locations that have the highest
average elevation is Colorad, Utah, and Wyoming.*/

/*Question 3*/
SELECT
    locationcategory.name 'Category Name',
    location.name 'Location',
    station.name 'Station',
    station.elevation 'Elevation'
FROM
	stationbylocation
JOIN
	station ON station.stationid = stationbylocation.staid
JOIN
	locationbycategory ON locationbycategory.locid = stationbylocation.locid
JOIN
	locationcategory ON locationcategory.lcid = locationbycategory.catid
JOIN
	location ON location.locationid = stationbylocation.locid
WHERE
	staid = (
	SELECT
		stationid
	FROM
		station
	ORDER BY
		elevation DESC
	LIMIT
		1);
/*This query list the location category name, the location name, the station name, and
elevation of the locations that include the one station in the entire database that has
the highest elevation. The station that has the highest elevation in the database is
Hanksville and it has the following location information:
State = Utah
County = Wayne County, UT
Zip Code = Hanksville, UT 84734
Climate Region = Southwest Climate Region
Country = United States*/

/*Question 4A*/
SELECT
    station.elevation AS Elevation,
    ROUND(AVG((tmin + tmax)/2), 2) 'Average Temperature',
    ABS(station.latitude) AS '|Latitude|'
FROM
	tminmax
LEFT JOIN
	station ON station.stationid = tminmax.stationid
WHERE
	tminmax.year >= 2008
GROUP BY
	tminmax.stationid
LIMIT
	50;
/*This query reports the station elevation, absolute value of the latitude, and the
average of the mean daily temperature measured at the station for the years 2008 and
beyond. Based on the results, it seems that if a station is further away from the
equator, then the average temperature of its station is lower than the average
temperature for all stations.*/

/*Question 4B*/
#1)
SELECT
	AVG(elevation) 'Average Elevation', AVG(ABS(latitude)) 'Average Absolute Latitude'
FROM
	(
    SELECT
        elevation 'Elevation',
		ABS(latitude) ' Latitude'
	FROM
		station
	WHERE
		station.stationid IN (
			SELECT
				tminmax.stationid
			FROM
				tminmax
			WHERE
				tminmax.year = 2008
			GROUP BY
				tminmax.stationid
    )
	ORDER BY
		elevation DESC
	LIMIT 50
    ) as stations_with_high_elevation;
/*This query gives me the average elevation and average absolute latitude for the
top 50 stations that have the highest elevation.*/

SELECT
	AVG(Mean_Daily_Temperature)
FROM (

	SELECT
		AVG((tmin + tmax)/ 2) Mean_Daily_Temperature
	FROM (

		SELECT
			station.stationid AS stationid
		FROM
			station
		WHERE
			station.stationid IN (
				SELECT
					tminmax.stationid
				FROM
					tminmax
				WHERE
					tminmax.year = 2008
				GROUP BY
					tminmax.stationid
		)
		ORDER BY
			elevation DESC
		LIMIT 50

) AS stationids_with_high_elevation
LEFT JOIN tminmax ON tminmax.stationid = stationids_with_high_elevation.stationid
GROUP BY
	stationids_with_high_elevation.stationid
) as mean_daily_temperature_top_50_stations_high_elevation;
/*This query gives me the average temperature for the top 50 highest elevated stations*/

#2)
SELECT
	AVG(elevation) 'Average Elevation', AVG(ABS(latitude)) 'Average Absolute Latitude'
FROM
	(
    SELECT
        elevation 'Elevation',
		ABS(latitude) ' Latitude'
	FROM
		station
	WHERE
		station.stationid IN (
			SELECT
				tminmax.stationid
			FROM
				tminmax
			WHERE
				tminmax.year = 2008
			GROUP BY
				tminmax.stationid
    )
	ORDER BY
		elevation
	LIMIT 50
    ) as stations_with_low_elevation;
/*This query gives me the average elevation and average absolute latitude for the
bottom 50 stations that have the lowest elevation.*/

SELECT
	AVG(Mean_Daily_Temperature) 'Average Temperature'
FROM (

	SELECT
		AVG((tmin + tmax)/ 2) Mean_Daily_Temperature
	FROM (

		SELECT
			station.stationid AS stationid
		FROM
			station
		WHERE
			station.stationid IN (
				SELECT
					tminmax.stationid
				FROM
					tminmax
				WHERE
					tminmax.year = 2008
				GROUP BY
					tminmax.stationid
		)
		ORDER BY
			elevation
		LIMIT 50

) AS stationids_with_low_elevation
LEFT JOIN tminmax ON tminmax.stationid = stationids_with_low_elevation.stationid
GROUP BY
	stationids_with_low_elevation.stationid
) as mean_daily_temperature_bottom_50_stations_low_elevation;
/*This query gives me the average temperature for the bottom 50 lowest elevated stations*/

#3)
SELECT
	AVG(elevation) 'Average Elevation', AVG(ABS(latitude)) 'Average Absolute Latitude'
FROM
	(
    SELECT
        elevation 'Elevation',
		ABS(latitude) ' Latitude'
	FROM
		station
	WHERE
		station.stationid IN (
			SELECT
				tminmax.stationid
			FROM
				tminmax
			WHERE
				tminmax.year = 2008
			GROUP BY
				tminmax.stationid
    )
	ORDER BY
		ABS(latitude)
	LIMIT 50
    ) as stations_with_low_latitude;
/*This query gives my the average elevation and average absolute latitude for the
bottom 50 stations that have the lowest absolute latitudes.*/

SELECT
	AVG(Mean_Daily_Temperature) 'Average Temperature'
FROM (

	SELECT
		AVG((tmin + tmax)/ 2) Mean_Daily_Temperature
	FROM (

		SELECT
			station.stationid AS stationid
		FROM
			station
		WHERE
			station.stationid IN (
				SELECT
					tminmax.stationid
				FROM
					tminmax
				WHERE
					tminmax.year = 2008
				GROUP BY
					tminmax.stationid
		)
		ORDER BY
			ABS(latitude)
		LIMIT 50

) AS stationids_with_low_latitude
LEFT JOIN tminmax ON tminmax.stationid = stationids_with_low_latitude.stationid
GROUP BY
	stationids_with_low_latitude.stationid
) as mean_daily_temperature_bottom_50_stations_low_altitude;
/*This query gives me the average temperature for the
bottom 50 stations that have the lowest absolute latitudes.*/

#4)
SELECT
	AVG(elevation) 'Average Elevation', AVG(ABS(latitude)) 'Average Absolute Latitude'
FROM
	(
    SELECT
        elevation 'Elevation',
		ABS(latitude) ' Latitude'
	FROM
		station
	WHERE
		station.stationid IN (
			SELECT
				tminmax.stationid
			FROM
				tminmax
			WHERE
				tminmax.year = 2008
			GROUP BY
				tminmax.stationid
    )
	ORDER BY
		ABS(latitude) DESC
	LIMIT 50
    ) as stations_with_high_latitude;
/*This query gives my the average elevation and average absolute latitude for the
top 50 stations that have the highest absolute latitudes.*/

SELECT
	AVG(Mean_Daily_Temperature) 'Average Temperature'
FROM (

	SELECT
		AVG((tmin + tmax)/ 2) Mean_Daily_Temperature
	FROM (

		SELECT
			station.stationid AS stationid
		FROM
			station
		WHERE
			station.stationid IN (
				SELECT
					tminmax.stationid
				FROM
					tminmax
				WHERE
					tminmax.year = 2008
				GROUP BY
					tminmax.stationid
		)
		ORDER BY
			ABS(latitude) DESC
		LIMIT 50

) AS stationids_with_high_latitude
LEFT JOIN tminmax ON tminmax.stationid = stationids_with_high_latitude.stationid
GROUP BY
	stationids_with_high_latitude.stationid
) as mean_daily_temperature_top_50_stations_high_latitude;

/*This query gives me the average temperature for the
top 50 stations that have the highest absolute latitudes.*/

/*Station Category	Average Elevation	Average Latitude	Average Temperature
High Elevation	 	3677.09	 			35.94	 			3.11
Low Elevation	 	-6.67	 			37.01	 			16.39
Low Latitudes	 	268.67	 			3.84	 			26.14
High Latitudes	 	89.93	 			71.80	 			-7.52*/

/*As it turns out the stations that have the highest average temperature are
the ones with the lowest latitudes*/

/*Question 5A*/
SELECT
	COUNT(*)
FROM (
	SELECT
		tminmax.stationid station_id,
        MAX(tminmax.year) max_year
	FROM
		tminmax
	WHERE
		tminmax.year >= 2000
	GROUP BY
		station_id
) AS tminmax_stationid_max_year
JOIN
	station ON station.stationid = tminmax_stationid_max_year.station_id
WHERE
	YEAR(station.maxdate) < max_year;
/*This query gives the number of stations for which the station's maxdate's year
is less than the maximum year in tminmax. There are 1427 stations that fit our
condition.*/

/*Question 5B*/
SELECT DISTINCT
	COUNT(*)
FROM
	tminmax
LEFT JOIN
	station ON station.stationid = tminmax.stationid
LEFT JOIN
	stationbylocation ON stationbylocation.staid = tminmax.stationid
LEFT JOIN
	location ON location.locationid = stationbylocation.locid
WHERE
	tminmax.year >= 2000 AND
    YEAR(location.maxdate) < YEAR(station.maxdate);
/*THere are no locations for which the location's maxdate's year is less than the
maximum year for any station in that location.*/

/*Question 6A*/
SELECT
	tminmax.year, ROUND(AVG((tmin + tmax) / 2), 2) average_temperature
FROM
	tminmax
WHERE
	stationid = 1115
GROUP BY
	tminmax.year;
/*This query gives the average yearly temperature for station
id 1115*/

SELECT
	ROUND(AVG(average_temp_for_year), 2) 'Average Yearly Temperature'
FROM
	(SELECT
		tminmax.year, AVG((tmin + tmax) / 2) average_temp_for_year
	FROM
		tminmax
	WHERE
		stationid = 1115
	GROUP BY
		tminmax.year) yearly_temps;
/*This query gives the average yearly temperature for station id 1115. It is about 9.80*/

SELECT
	AVG(ABS(average_temp_for_year - 9.80)) 'Amplitude'
FROM
	(SELECT
		tminmax.year, AVG((tmin + tmax) / 2) average_temp_for_year
	FROM
		tminmax
	WHERE
		stationid = 1115
	GROUP BY
		tminmax.year) yearly_temps;
/*This query gives the amplitude. It is about 0.37*/

/*Question 6B*/
SELECT
	AVG((tmin + tmax)/2) 'Mean Daily Temperature Averaged Over the Years 2008 to Present'
FROM
	tminmax
WHERE
	stationid = 1115 AND
    year >= 2008;
/*This query gets the mean daily temperature averaged over the years 2008 to present for
station id 1115. It is about 10.09*/

SELECT
	*
FROM
	tminmax
WHERE
	stationid = 1115;
/*This query gets the daily temperature data from the tminmax table for station id 1115.*/

SELECT
	tminmax.year year, MIN(tmin)  minimum_temperature_for_year
FROM
	tminmax
WHERE
	stationid = 1115 AND
	tmin < 10.09
GROUP BY
	tminmax.year;
/*This query gets the minimum temperature for each year for station 1115 where tmin is less than 10.09
(mean daily temperature averaged over the years 2008 to present for station id 1115)*/

SELECT
	AVG(st1115.dayofyear) 'Phi'
FROM
	(SELECT
		*
	FROM
		tminmax
	WHERE
		stationid = 1115) st1115
JOIN
	(SELECT
		tminmax.year year, MIN(tmin)  minimum_temperature_for_year
	FROM
		tminmax
	WHERE
		stationid = 1115 AND
		tmin < 10.09
	GROUP BY
		tminmax.year) min_temp_year ON min_temp_year.year = st1115.year
WHERE
	st1115.tmin = min_temp_year.minimum_temperature_for_year;
/*This query gets the average day of the year when the temperature is at its lowest. It is about 181.9456*/
