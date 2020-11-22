CREATE TABLE IF NOT EXISTS reviews_data (
    id                  SERIAL NOT NULL PRIMARY KEY,
    scraper_date        TIMESTAMP not NULL,

    -- Google or Apple
    review_source       text,
    -- Name of App
    review_app          text,
    -- PT or EN
    review_language     text,
    review_raw          TEXT not NULL,
    review_formatted    text,
    
    -- O default -1 significa que o review ainda não foi avaliado
    -- Se for 0, quer dizer que não é de acessibilidade.
    -- Caso seja 1, é de acessibilidade.
    is_a11y_human       INTEGER DEFAULT -1, 
    is_a11y_machine     INTEGER DEFAULT -1

);


-- A(eleven)y
CREATE TABLE IF NOT EXISTS a11y_words (
    -- Accessibility (a11y)

    id                  SERIAL NOT NULL PRIMARY KEY,
    reviews_data_id     INTEGER not NULL REFERENCES reviews_data(id),
    word                text not NULL    
);