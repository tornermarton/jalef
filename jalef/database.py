import mysql.connector


class Cursor(object):
    def __init__(self, connection):
        self._connection = connection

        self._cursor = None

    def __enter__(self):
        if self._cursor is None:
            self._cursor = self._connection.cursor()
        else:
            raise ValueError("Cursor is initialized!")

        return self._cursor

    def __exit__(self, exc_type, exc_value, traceback):
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None
        else:
            raise ValueError("Cursor is not initialized!")

        return self


class DatabaseConnection(object):
    def __init__(self, host, user, password, database):
        self.__host = host
        self.__user = user
        self.__password = password
        self.__database = database

        self._connection = None

    def __enter__(self):
        self.connect()

        return self._connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

        return self

    def connect(self):
        self._connection = mysql.connector.connect(
            host=self.__host,
            user=self.__user,
            passwd=self.__password,
            database=self.__database
        )

    def disconnect(self):
        self._connection.close()
        self._connection = None
