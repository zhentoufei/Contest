-- drop table if exists tmp1;
-- create table tmp1
-- as select a.*,b.prediction_score as gbdt100
-- from wepon_datalabel16 a join wepon_16_gbdt100 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- select count(*) from tmp1;

-- drop table if exists tmp2;
-- create table tmp2
-- as select a.*,b.prediction_score as gbdt200
-- from tmp1 a join wepon_16_gbdt200 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- select count(*) from tmp1;

-- drop table if exists tmp3;
-- create table tmp3
-- as select a.*,b.prediction_score as gbdt300
-- from tmp2 a join wepon_16_gbdt300 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;


-- drop table if exists tmp4;
-- create table tmp4
-- as select a.*,b.prediction_score as gbdt400
-- from tmp3 a join wepon_16_gbdt400 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp5;
-- create table tmp5
-- as select a.*,b.prediction_score as gbdt500
-- from tmp4 a join wepon_16_gbdt500 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp6;
-- create table tmp6
-- as select a.*,b.prediction_score as gbdt600
-- from tmp5 a join wepon_16_gbdt600 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp7;
-- create table tmp7
-- as select a.*,b.prediction_score as gbdt700
-- from tmp6 a join wepon_16_gbdt700 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp8;
-- create table tmp8
-- as select a.*,b.prediction_score as gbdt800
-- from tmp7 a join wepon_16_gbdt800 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp9;
-- create table tmp9
-- as select a.*,b.prediction_score as gbdt900
-- from tmp8 a join wepon_16_gbdt900 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp10;
-- create table tmp10
-- as select a.*,b.prediction_score as gbdt1000
-- from tmp9 a join wepon_16_gbdt1000 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp11;
-- create table tmp11
-- as select a.*,b.prediction_score as rf64
-- from tmp10 a join wepon_16_rf64 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp12;
-- create table tmp12
-- as select a.*,b.prediction_score as rf128
-- from tmp11 a join wepon_16_rf128 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp13;
-- create table tmp13
-- as select a.*,b.prediction_score as rf256
-- from tmp12 a join wepon_16_rf256 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp14;
-- create table tmp14
-- as select a.*,b.prediction_score as rf320
-- from tmp13 a join wepon_16_rf320 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp15;
-- create table tmp15
-- as select a.*,b.prediction_score as rf384
-- from tmp14 a join wepon_16_rf384 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp16;
-- create table tmp16
-- as select a.*,b.prediction_score as rf448
-- from tmp15 a join wepon_16_rf448 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp17;
-- create table tmp17
-- as select a.*,b.prediction_score as rf512
-- from tmp16 a join wepon_16_rf512 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists tmp18;
-- create table tmp18
-- as select a.*,b.prediction_score as rf576
-- from tmp17 a join wepon_16_rf576 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- drop table if exists wepon_stack16;
-- create table wepon_stack16
-- as select a.*,b.prediction_score as rf640
-- from tmp18 a join wepon_16_rf640 b
-- on a.user_id=b.user_id and a.item_id=b.item_id;

-- --取wepon_16gbdt所确定的前20万个ui对
-- drop table if exists wepon_stack16_20w;
-- create table wepon_stack16_20w
-- as select d.* from 
-- wepon_stack16 d
-- join
-- (
--     select user_id,item_id
--     from
--     (
--         select user_id,item_id,pred,row_number() over(partition by partition_column order by pred desc) as rank 
--         from
--         (
--             select user_id,item_id,sum(prediction_score) as pred,sum(0+0) as partition_column
--             from wepon_16gbdt
--             group by user_id,item_id
--         )a
--     )b
--     where rank<=200000
-- )c
-- on d.user_id=c.user_id and d.item_id=c.item_id;

drop table if exists tmp1;
create table tmp1
as select a.*,b.prediction_score as meta_rf1
from wepon16 a join 16pred1 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp1;

drop table if exists tmp2;
create table tmp2
as select a.*,b.prediction_score as meta_gbdt1
from tmp1 a join 16pred2 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp2;


drop table if exists tmp3;
create table tmp3
as select a.*,b.prediction_score as meta_gbdt2
from tmp2 a join 16pred3 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp3;

drop table if exists tmp4;
create table tmp4
as select a.*,b.prediction_score as meta_gbdt3
from tmp3 a join 16pred4 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp4;

drop table if exists tmp5;
create table tmp5
as select a.*,b.prediction_score as meta_rf2
from tmp4 a join 16pred5 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp5;


drop table if exists tmp6;
create table tmp6
as select a.*,b.prediction_score as meta_rf3
from tmp5 a join 16pred6 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp6;


drop table if exists tmp7;
create table tmp7
as select a.*,b.prediction_score as meta_rf4
from tmp6 a join 16pred7 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp7;



drop table if exists tmp8;
create table tmp8
as select a.*,b.prediction_score as meta_gbdt4
from tmp7 a join 16pred8 b
on a.user_id=b.user_id and a.item_id=b.item_id;

select count(*) from tmp8;