from ._http_manager import _HTTPManager


class HTTP(_HTTPManager):
    def create_internal_transfer(self, **kwargs):
        """
        Create internal transfer.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/account_asset/#t-createinternaltransfer.
        :returns: Request results as dictionary.
        """

        suffix = "/asset/v1/private/transfer"
        if self._verify_string(kwargs, "amount"):
            return self._submit_request(
                method="POST",
                path=self.endpoint + suffix,
                query=kwargs,
                auth=True
            )
        else:
            self.logger.error("amount must be in string format")

    def create_subaccount_transfer(self, **kwargs):
        """
        Create internal transfer.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/account_asset/#t-createsubaccounttransfer.
        :returns: Request results as dictionary.
        """

        suffix = "/asset/v1/private/sub-member/transfer"

        if self._verify_string(kwargs, "amount"):
            return self._submit_request(
                method="POST",
                path=self.endpoint + suffix,
                query=kwargs,
                auth=True
            )
        else:
            self.logger.error("amount must be in string format")

    def query_transfer_list(self, **kwargs):
        """
        Create internal transfer.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/account_asset/#t-querytransferlist.
        :returns: Request results as dictionary.
        """

        suffix = "/asset/v1/private/transfer/list"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    def query_subaccount_list(self):
        """
        Create internal transfer.

        :returns: Request results as dictionary.
        """

        suffix = "/asset/v1/private/sub-member/member-ids"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query={},
            auth=True
        )

    def query_subaccount_transfer_list(self, **kwargs):
        """
        Create internal transfer.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/account_asset/#t-querysubaccounttransferlist.
        :returns: Request results as dictionary.
        """

        suffix = "/asset/v1/private/sub-member/transfer/list"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
