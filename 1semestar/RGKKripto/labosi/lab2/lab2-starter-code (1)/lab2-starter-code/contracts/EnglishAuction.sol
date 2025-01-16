// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Auction.sol";

contract EnglishAuction is Auction {

    uint internal highestBid;
    uint internal initialPrice;
    uint internal biddingPeriod;
    uint internal lastBidTimestamp;
    uint internal minimumPriceIncrement;

    address internal highestBidder;

    constructor(
        address _sellerAddress,
        address _judgeAddress,
        Timer _timer,
        uint _initialPrice,
        uint _biddingPeriod,
        uint _minimumPriceIncrement
    ) Auction(_sellerAddress, _judgeAddress, _timer) {
        initialPrice = _initialPrice;
        biddingPeriod = _biddingPeriod;
        minimumPriceIncrement = _minimumPriceIncrement;

        // Start the auction at contract creation.
        lastBidTimestamp = time();
    }

    function bid() public payable {
        // TODO Your code here
        //revert("Not yet implemented");
        require(outcome == Outcome.NOT_FINISHED, "Outcome is finnished!");
        bool status = (lastBidTimestamp + biddingPeriod) > time();
        if(!status){
            if(highestBid >= initialPrice){
                outcome = Outcome.SUCCESSFUL;
            }
            else{
                outcome = Outcome.NOT_SUCCESSFUL;
            }
            require(status, "Bidding has ended");
        }
        if (highestBid == 0){
            require(msg.value >= initialPrice, "Bidding value has to be initial price or higher");
            highestBid = msg.value;
            highestBidder = msg.sender;
            lastBidTimestamp = time();
        }else{
            require(msg.value >= highestBid+minimumPriceIncrement, "Bid isn't much bigger from current highest bid");
            (bool status, ) = highestBidder.call{value: highestBid}("");
            require(status, "Ether not sent");
            highestBid = msg.value;
            highestBidder = msg.sender;
            lastBidTimestamp = time();
        }
    }

    function getHighestBidder() override public returns (address) {
        // TODO Your code here
        //revert("Not yet implemented");
        bool status = (lastBidTimestamp + biddingPeriod) > time();
        if(!status){
            if(highestBid >= initialPrice){
                outcome = Outcome.SUCCESSFUL;
            }
            else{
                outcome = Outcome.NOT_SUCCESSFUL;
            }
        }
        if (outcome != Outcome.SUCCESSFUL){
            return address(0);
        }
        else{
            return highestBidder;
        }
    }

    function enableRefunds() public {
        // TODO Your code here
        //revert("Not yet implemented")
        require(outcome != Outcome.SUCCESSFUL,"Cannot refund successful auction");
        outcome = Outcome.NOT_SUCCESSFUL;
    }

}