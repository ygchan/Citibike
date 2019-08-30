-- This project idea is sparked from a friend.
-- Inside of only using the sample database provided by textbook to practice 
-- SQL. It might be a good idea to build a relational database warehouse
-- using SQL and Python.

-- 01. First let's create a SQL schema statement
-- This is for the trip_history data
create table trip_history
(
   trip_duration smallint unsigned,
   trip_id int unsigned,
   -- Full time: yyyy-mm-dd hh:mi:ss
   start_time datetime,
   stop_time datetime,
   -- Station_ID are foreign keys
   -- Need to read how to setup contrain
   start_station_id varchar(20),
   end_station_id varchar(20),
   bike_id varchar(20),
   -- Annual subscription, Visitor Day/Weekly Passes
   user_type enum('A', 'V'),
   birth_year char(4),
   -- Male, Female, Unknown
   gender enum('M', 'F', 'U'),
   constraint pk_trip primary key(trip_id),
   constraint fk_s_station_id foreign key (start_station_id) references station (station_id),
   constraint fk_e_station_id foreign key (end_station_id) references station (station_id)
);

-- 02. Because the startion id, name, geo coordinates are repeated.
-- I would like to normalize them and store them into a separate table.
create table station
(
   station_id smallint unsigned,
   longitude float,
   lattitude float
   constraint pk_station primary key(station_id)
);
